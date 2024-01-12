from torch.utils.data import Dataset
from utils.vocab import LabelVocab
from utils.evaluator import Evaluator
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
import re
import json


class SLU(Dataset):
    label_vocab = LabelVocab(r'..\Project\data')
    evaluator = Evaluator()
    def __init__(self, root, data_file, test=False):
        super(SLU, self).__init__()
        self.data = self.load_data(data_file, test)
        
        
    
    def load_data(self, data_file, test=False):
        dataset = json.load(open(data_file, 'r'))
        examples = []
        def new_data(ex):
            
            utt = ex['asr_1best'] if test else ex['manual_transcript']
            
            input =  tokenizer(utt, return_tensors='pt', return_offsets_mapping=True)# 表示begin和end，占位用，没有其他意义
            slots = {}
            for label in ex['semantic']:
                act_slot = f'{label[0]}-{label[1]}'
                if len(label) == 3:
                    slots[act_slot] = label[2]
            tags = ['O'] * (input.input_ids.shape[-1])
            tmp = ''
            # 找原来的utt在新的utt（tmp变量）中的位置
            for idx in input.offset_mapping.squeeze(0):
                if idx[1] - idx[0] == 1:
                     tmp += utt[idx[0]]
                else:
                    tmp += 'o'
            
            for slot in slots:
                value = slots[slot]
                bidx = tmp.find(value)
                if bidx != -1:
                    tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                    tags[bidx] = f'B-{slot}'
            l = SLU.label_vocab
            tag_id = [l.convert_tag_to_idx(tag) for tag in tags]
            slotvalue = [f'{slot}-{value}' for slot, value in slots.items()]
            assert len(tmp) == len(tag_id), 'utt长度与tag_id不匹配！'
            return {'utt': utt ,  'tag_id':tag_id, 'slotvalue':slotvalue}
        for data in dataset:
            for utt in data:
                ex = new_data(utt)
                examples.append(ex)
        return examples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
        