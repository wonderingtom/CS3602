#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

from utils.initialization import *

MASK_VALUE = -2 ** 32 + 1


class MultiLevelWordAdapter(nn.Module):

    def __init__(self, config):
        super(MultiLevelWordAdapter, self).__init__()
        self.config = config
        self.device = set_torch_device(config.device)
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        
        self.char_encoder = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.word_encoder = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        
        self.char_attention = nn.MultiheadAttention(config.hidden_size, num_heads=4, batch_first=True)
        self.word_attention = nn.MultiheadAttention(config.hidden_size, num_heads=4, batch_first=True)

        self.char_sent_attention = MLPAttention(config.hidden_size * 2, config.dropout)
        self.word_sent_attention = MLPAttention(config.hidden_size * 2, config.dropout)

        self.word_adapter = WordAdapter(config.hidden_size * 2)

        decoder_in_dim = config.hidden_size * 2 + config.hidden_size * 2
        self.char_decoder = getattr(nn, self.cell)(decoder_in_dim, config.hidden_size * 2, num_layers=config.num_layer, bidirectional=False, batch_first=True)
        self.word_decoder = getattr(nn, self.cell)(decoder_in_dim, config.hidden_size, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size * 2, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        char_text = batch.input_ids
        word_text = batch.input_ids_word
        lengths_char = batch.lengths
        lengths_word = batch.lengths_word
        w2c_idx_matrix = batch.padded_idx_mtx  # (B, char_seq_len, word_seq_len)

        char_embed = self.word_embed(char_text)
        word_embed = self.word_embed(word_text)
        packed_char_embed = rnn_utils.pack_padded_sequence(char_embed, lengths_char, batch_first=True, enforce_sorted=True)
        packed_word_embed = rnn_utils.pack_padded_sequence(word_embed, lengths_word, batch_first=True, enforce_sorted=False)
        
        packed_rnn_out_char, h_t_c_t = self.char_encoder(packed_char_embed)  # bsize x seqlen x dim
        packed_rnn_out_word, h_t_c_t = self.char_encoder(packed_word_embed)  # bsize x seqlen x dim
        char_rnn_hiddens, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out_char, batch_first=True)
        char_rnn_hiddens = self.dropout_layer(char_rnn_hiddens)
        key_padding_mask_char = torch.zeros((len(unpacked_len), max(unpacked_len)), dtype=torch.bool, device=self.device)
        for i, length in enumerate(unpacked_len):
            key_padding_mask_char[i, length:] = True
        char_attention_hiddens, _ = self.char_attention(char_rnn_hiddens, char_rnn_hiddens, char_rnn_hiddens, key_padding_mask=key_padding_mask_char)
        char_hiddens = torch.cat([char_rnn_hiddens, char_attention_hiddens], dim=-1)

        word_rnn_hiddens, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out_word, batch_first=True)
        word_rnn_hiddens = self.dropout_layer(word_rnn_hiddens)
        key_padding_mask_word = torch.zeros((len(unpacked_len), max(unpacked_len)), dtype=torch.bool, device=self.device)
        for i, length in enumerate(unpacked_len):
            key_padding_mask_word[i, length:] = True
        word_attention_hiddens, _ = self.char_attention(word_rnn_hiddens, word_rnn_hiddens, word_rnn_hiddens, key_padding_mask=key_padding_mask_word)
        word_hiddens = torch.cat([word_rnn_hiddens, word_attention_hiddens], dim=-1)

        intent_char_sent = self.char_sent_attention(char_hiddens, rmask=1.0*(~key_padding_mask_char))
        intent_word_sent = self.word_sent_attention(word_hiddens, rmask=1.0*(~key_padding_mask_word))

        intent_hidden = self.word_adapter(intent_char_sent, intent_word_sent)

        char_decoder_input = torch.cat([char_hiddens, intent_hidden.unsqueeze(1).repeat(1, char_hiddens.shape[1], 1)], dim=-1)
        word_decoder_input = torch.cat([word_hiddens, intent_hidden.unsqueeze(1).repeat(1, word_hiddens.shape[1], 1)], dim=-1)

        packed_char_decoder_input = rnn_utils.pack_padded_sequence(char_decoder_input, lengths_char, batch_first=True, enforce_sorted=True)
        packed_word_decoder_input = rnn_utils.pack_padded_sequence(word_decoder_input, lengths_word, batch_first=True, enforce_sorted=False)

        packed_char_lstm_output, h_t_c_t = self.char_decoder(packed_char_decoder_input)
        packed_word_lstm_output, h_t_c_t = self.word_decoder(packed_word_decoder_input)

        char_lstm_output, unpacked_len = rnn_utils.pad_packed_sequence(packed_char_lstm_output, batch_first=True)
        word_lstm_output, unpacked_len = rnn_utils.pad_packed_sequence(packed_word_lstm_output, batch_first=True)

        aligned_word_lstm_output = w2c_idx_matrix @ word_lstm_output
        slot_hiddens = self.word_adapter(char_lstm_output, aligned_word_lstm_output)

        hiddens = self.dropout_layer(slot_hiddens)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )


class MLPAttention(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(MLPAttention, self).__init__()

        # Record parameters
        self.__input_dim = input_dim
        self.__dropout_rate = dropout_rate

        # Define network structures
        self.__dropout_layer = nn.Dropout(self.__dropout_rate)
        self.__sent_attention = nn.Linear(self.__input_dim, 1, bias=False)

    def forward(self, encoded_hiddens, rmask=None):
        """
        Merge a sequence of word representations as a sentence representation.
        :param encoded_hiddens: a tensor with shape of [bs, max_len, dim]
        :param rmask: a mask tensor with shape of [bs, max_len]
        :return:
        """
        # TODO: Do dropout ?
        dropout_input = self.__dropout_layer(encoded_hiddens)
        score_tensor = self.__sent_attention(dropout_input).squeeze(-1)

        if rmask is not None:
            assert score_tensor.shape == rmask.shape, "{} vs {}".format(score_tensor.shape, rmask.shape)
            score_tensor = rmask * score_tensor + (1 - rmask) * MASK_VALUE

        score_tensor = F.softmax(score_tensor, dim=-1)
        # matrix multiplication: [bs, 1, max_len] * [bs, max_len, dim] => [bs, 1, dim]
        sent_output = torch.matmul(score_tensor.unsqueeze(1), dropout_input).squeeze(1)

        return sent_output
    
class WordAdapter(nn.Module):
    def __init__(self, input_dim):
        super(WordAdapter, self).__init__()

        self.fc = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, char, word):
        # char, word: (B, *, E)

        # lambda: (B, *)
        lambda_tmp = torch.sigmoid(torch.sum(char * self.fc(word), dim=-1))
        lambda_tmp = lambda_tmp.unsqueeze(-1)
        output = (1 - lambda_tmp) * char + lambda_tmp * word

        return output