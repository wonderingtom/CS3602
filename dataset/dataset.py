import json
import re
import os
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
# from transformers import AutoTokenizer, AutoModel


from utils.vocab import Label, LabelVocab_history



@dataclass
class Sentence:
    vector_with_noise: torch.Tensor
    vector_without_noise: torch.Tensor
    tokens_with_noise: List[str]
    tokens_without_noise: List[str]


class get_dataset(Dataset):
    pattern = re.compile(r'\(.*\)')

    def __init__(self, data_path, label_converter: LabelVocab_history, dataset_path, args):
        self.device = torch.device("cuda:%d" % args.device)
        if os.path.isfile(dataset_path):
            self._data = torch.load(dataset_path, map_location=self.device)
            return
        self.label_converter = label_converter
        # tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
        # model = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext')
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
        # output_len = list(model.parameters())[-1].shape[0]

        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
        self._data = []
        for i in data:
            n_utt = len(i)
            utt_vector = [None] * n_utt
            labels = [None] * n_utt
            for utt in i:
                text_with_noise = utt['asr_1best']
                text_without_noise = utt['manual_transcript']
                # remove special tokens in the dataset, i.e., (unknown)
                text_with_noise = re.sub(self.pattern, '', text_with_noise)
                text_without_noise = re.sub(self.pattern, '', text_without_noise)
                k = utt['utt_id'] - 1
                model_input = tokenizer(text_with_noise, return_tensors='pt')
                output = model(**model_input)
                vector_with_noise = output[0][0].to(self.device).detach()
                model_input = tokenizer(text_without_noise, return_tensors='pt')
                output = model(**model_input)
                vector_without_noise = output[0][0].to(self.device).detach()
                tokens_with_noise = tokenizer.tokenize(text_with_noise)
                tokens_without_noise = tokenizer.tokenize(text_without_noise)
                utt_vector[k] = Sentence(vector_with_noise, vector_without_noise, tokens_with_noise, tokens_without_noise)
                bio_labels = self._get_bio_labels(tokens_without_noise, utt['semantic'])
                # bio_labels = self._get_bio_labels(tokens_with_noise, utt['semantic'])
                tensor = torch.zeros([len(bio_labels), self.label_converter.num_tags])
                for i2, v2 in enumerate(bio_labels):
                    tensor[i2, v2] = 1
                labels[k] = tensor.to(self.device)
            self._data.append((utt_vector, labels))
        torch.save(self._data, dataset_path)

    def _get_bio_labels(self, text: List[str], labels: List[Tuple[str, str, str]]) -> List[int]:
        labels = [i for i in labels]

        # make labels appear in the same order as they appear in the text
        labels = self._rearrange_labels(''.join(text), labels)

        ret = [self.label_converter.convert_tag_to_idx(Label(2, '', ''))] * (len(text) + 2)
        if len(labels) == 0:
            return ret
        j = 0
        act, slot, value = labels[j]
        begin = True
        for i, v in enumerate(text):
            if value.startswith(v):
                bio = 0 if begin else 1
                ret[i + 1] = self.label_converter.convert_tag_to_idx(Label(bio, act, slot))
                begin = False
                value = value[len(v):]
            if len(value) == 0:
                j += 1
                if j >= len(labels):
                    break
                act, slot, value = labels[j]
                begin = True
        if j < len(labels):
            raise RuntimeError('The text and tags are inconsistent.')
        return ret

    @staticmethod
    def _rearrange_labels(text: str, labels: List[Tuple[str, str, str]]):
        labels = sorted(labels, key=lambda x: len(x[2]), reverse=True)
        for i, v in enumerate(labels):
            j = text.find(v[2])
            labels[i].append(j)
            text = text.replace(v[2], '#' * len(v[2]), 1)
        labels = [i for i in labels if i[3] != -1]
        labels = sorted(labels, key=lambda x: x[3])
        labels = [i[:3] for i in labels]
        return labels

    def __getitem__(self, index: int) -> Tuple[List[Sentence], List[torch.Tensor]]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)
