import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import jieba

jieba.load_userdict("data\lexicon\poi_name.txt")

class SlotGatedSLU(nn.Module):

    def __init__(self, config):
        super(SlotGatedSLU, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.slot_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=4, batch_first=True)
        self.intent_attention = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=4, batch_first=True)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)

        self.intent_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.slot_gate_v = nn.Parameter(torch.Tensor(config.hidden_size))

        self.output_layer = TaggingFNNDecoder(config.hidden_size * 2, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
       
        hiddens = self.dropout_layer(rnn_out)

        padding_mask = (tag_mask == 0)

        slot_output, _ = self.slot_attention(hiddens, hiddens, hiddens, key_padding_mask=padding_mask)
        slot_output = hiddens
        intent_output, _ = self.intent_attention(hiddens, hiddens, hiddens, key_padding_mask=padding_mask)
        intent_output = torch.cat([intent_output[:, 0], intent_output[range(len(unpacked_len)), unpacked_len-1]], dim=1).unsqueeze(1)

        slot_gate = torch.tanh(slot_output + self.intent_proj(intent_output))

        slot_gate = torch.sum(slot_gate, dim=-1)

        tag_input = torch.cat([hiddens, slot_output * slot_gate.unsqueeze(2)], dim=-1)

        tag_output = self.output_layer(tag_input, tag_mask, tag_ids)

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
                    idx_buff = self.refine_value(batch.utt[i], idx_buff)
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
                idx_buff = self.refine_value(batch.utt[i], idx_buff)
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            # 消除内部空格
            pred_tuple = [item.replace(" ", "") for item in pred_tuple]
            # 去重
            pred_tuple = list(set(pred_tuple))
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()
        
    def refine_value(self, utt, original_list):
        # 对utt进行jieba分词
        seg_list = list(jieba.cut(utt, cut_all=False))
        # 初始化一个空的列表，用于存储索引列表
        index_list = []
        # 当前索引
        current_index = 0
        # 遍历分词结果
        for seg in seg_list:
            # 获取当前分词的长度
            seg_length = len(seg)
            # 生成当前分词的索引列表
            seg_indexes = [i for i in range(current_index, current_index + seg_length)]
            # 将索引列表添加到结果中
            index_list.append(seg_indexes)
            # 更新当前索引
            current_index += seg_length
        for index in original_list:
            for sublist in index_list:
                if index in sublist:
                    original_list.extend(sublist)
                    original_list = list(set(original_list))
                    break
        return sorted(original_list)

        

        
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