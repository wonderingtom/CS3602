import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer
import math

class TransformerCRF(nn.Module):

    def __init__(self, config,
                 nhead=4,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=2048,
                 has_pos=False,
                 ):
        super(TransformerCRF, self).__init__()
        self.config = config

        self.has_pos = has_pos
        if has_pos:
            self.pos_encoder = PositionalEncoding(config.hidden_size)

        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.project_layer = nn.Linear(config.embed_size, config.hidden_size)

        encoder_layer = TransformerEncoderLayer(config.hidden_size, nhead, dim_feedforward, config.dropout, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(config.hidden_size, nhead, dim_feedforward, config.dropout, batch_first=True)
        decoder_norm = nn.LayerNorm(config.hidden_size)

        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.dropout_layer = nn.Dropout(p=config.dropout)
        # self.label_ffn = nn.Linear(config.hidden_size, config.num_tags)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        # input_ids_asr = batch.input_ids_asr
        # lengths_asr = batch.lengths_asr

        # src = self.project_layer(self.word_embed(input_ids_asr))
        tgt = self.project_layer(self.word_embed(input_ids))

        if self.has_pos:
            # src = self.pos_encoder(src)
            tgt = self.pos_encoder(tgt)
        else:
            pos_embed = None

        # max_len = max(lengths_asr)
        # src_key_padding_mask = torch.zeros((len(lengths_asr), max_len), dtype=torch.bool)
        # for i, length in enumerate(lengths_asr):
        #     src_key_padding_mask[i, length:] = 1

        max_len = max(lengths)
        tgt_key_padding_mask = torch.zeros((len(lengths), max_len), dtype=torch.bool)
        for i, length in enumerate(lengths):
            tgt_key_padding_mask[i, length:] = 1
       
        memory = self.encoder(tgt, src_key_padding_mask=tgt_key_padding_mask.to(tgt.device))

        # hs = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask.to(src.device), memory_key_padding_mask=src_key_padding_mask.to(src.device))

        hiddens = self.dropout_layer(memory)
        
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



class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        """
        :param label_vocab: Dict: 每个label对应的idx，例如{"O": 0, "B-PER": 1, ...}
        :param bioes: bool: 是bioes形式的标注还是bio形式的标注，默认bio
        整个初始化过程其实就是创建了一个状态转移矩阵transition
        """
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2                         # 增加了<sos>和<eos>
        self.bioes = bioes

        self.start = self.label_size - 2                               # 倒数第2个label是<sos>
        self.end = self.label_size - 1                                 # 倒数第1个label是<eos>
        transition = torch.randn(self.label_size, self.label_size)     # 初始化一个(num_labels+2, num_labels+2)的矩阵
        self.transition = nn.Parameter(transition)                     # 将状态转移矩阵转化为可训练参数
        self.initialize()

    def initialize(self):
        """
        对转移矩阵进行进一步操作，将所有必然不可达的状态都设置为一个默认值-100
        注意第一个axis是to_label, 第二个axis才是from_label
        """
        self.transition.data[:, self.end] = -100.0                     # <eos>不可以向任何一个label转移
        self.transition.data[self.start, :] = -100.0                   # 没有任何一个label可以转移到<sos>

        # 对num_labels两层遍历，排除所有不合理的情况
        for label, label_idx in self.label_vocab.items():              # ("O": 0), ("B-PER": 1), ...
            if label.startswith('I-') or label.startswith('E-'):       # <sos>不能跳过B直接转移到I和E
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):       # <eos>不能由B或I转移得到（这是BIOES的规则）
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    # 1. 如果是BIOES形式，则
                    # 1) [O, E, S]中的任意一个状态，都可以转移到[O, B, S]中任意一个状态，不论前后两个状态的label是否相同
                    # - 例如，可以从E-PER转移到B-LOC
                    # 2) 当label相同时，允许B->I, B->E, I->I, I->E
                    
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    # 2. 如果是BIO形式，则
                    # 1) 任何一个状态都可能转移到B和O
                    # 2) I状态只能由相同label的B或者I得到
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[label_to_idx, label_from_idx] = -100.0
        
    @staticmethod
    def pad_logits(logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()                     # (batch, seq_len, num_labels)
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,   
                               requires_grad=False)                # 返回一个形状为(batch, seq_len, 2)的tensor，所有位置填充为-100
        logits = torch.cat([logits, pads], dim=2)                  # 拼接得到(batch, seq_len, num_labels+2)
        return logits








class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
