#coding=utf8
import torch
import torch.nn as nn
from transformers import AutoModel
from TorchCRF import CRF

class BERT(nn.Module):

    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = FNN(config.hidden_size, config.num_tags, config.tag_pad_idx, config.use_crf)

    def forward(self, batch):
        tag_ids = batch[1].to(self.config.device)
        input = batch[0].to(self.config.device)
        input.input_ids
        hiddens = self.bert(**input).last_hidden_state
        hiddens = self.dropout_layer(hiddens)
        tag_output = self.output_layer(hiddens, input.attention_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = batch[1].shape[0]
        labels = batch[3]  # 实际SLU任务的label
        output = self.forward(batch)
        prob = output[0].view(batch_size, -1, label_vocab.num_tags)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[1:1+len(batch[2][i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch[2][i][j] for j in idx_buff])
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
                value = ''.join([batch[2][i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class FNN(nn.Module):

    def __init__(self, input_size, num_tags, pad_id, use_crf=False):
        super(FNN, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.use_crf = use_crf
        if self.use_crf:
            self.crf = CRF(num_tags=self.num_tags, batch_first=True)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens.view(-1, hiddens.shape[-1]))
        # mask = mask.view(-1, hiddens.shape[-1])
        # logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            if self.use_crf:
                loss = self.crf(logits.view(labels.shape[0], -1, logits.shape[-1]), labels, reduction='mean')
                loss = -1 * loss
            else:
                loss = self.loss_fct(logits, labels.view(-1))
            return prob, loss
        return (prob, )
