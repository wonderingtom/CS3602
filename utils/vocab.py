#coding=utf8
import os, json, jieba
import torch
from dataclasses import dataclass
from typing import Iterator, List, Tuple

PAD = '<pad>'
UNK = '<unk>'
BOS = '<s>'
EOS = '</s>'

@dataclass
class Label:
    bio: int
    act: str
    slot: str


@dataclass
class Sentence:
    vector_with_noise: torch.Tensor
    vector_without_noise: torch.Tensor
    tokens_with_noise: List[str]
    tokens_without_noise: List[str]

class Vocab():

    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['manual_transcript']
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])

class LabelVocab_history:
    def __init__(self, ontology_path):
        with open(ontology_path, encoding='utf-8') as f:
            data = json.load(f)
        self.act_to_index = {v: i for i, v in enumerate(data['acts'])}
        self.index_to_act = {i: v for i, v in enumerate(data['acts'])}
        self.slot_to_index = {v: i for i, v in enumerate(data['slots'])}
        self.index_to_slot = {i: v for i, v in enumerate(data['slots'])}
        self.n_acts = len(self.act_to_index)
        self.n_slots = len(self.slot_to_index)

    def convert_tag_to_idx(self, label: Label) -> int:
        if label.bio == 2:
            return 2 * self.n_acts * self.n_slots
        i = 0 if label.bio == 0 else 1
        i = i * self.n_acts + self.act_to_index[label.act]
        i = i * self.n_slots + self.slot_to_index[label.slot]
        return i

    def convert_idx_to_tag(self, i: int) -> Label:
        if i == 2 * self.n_acts * self.n_slots:
            return Label(2, '', '')
        slot_idx = i % self.n_slots
        i //= self.n_slots
        act_idx = i % self.n_acts
        i //= self.n_acts
        bio = 0 if i == 0 else 1
        return Label(bio, self.index_to_act[act_idx], self.index_to_slot[slot_idx])

    @property
    def num_tags(self) -> int:
        return 2 * self.n_acts * self.n_slots + 1

class LabelVocab():

    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r'))
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)

class Alphabet():

    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Alphabet, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='UTF-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['manual_transcript']
                words = jieba.lcut(text, cut_all=True)
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK])
