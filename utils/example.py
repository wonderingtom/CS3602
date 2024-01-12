import json
import jieba
from utils.vocab import Vocab, LabelVocab, Alphabet
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
import numpy as np
class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}')
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did

        self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        res = 0
class Dataloader():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, train=False):
        dataset = json.load(open(data_path, 'r', encoding='UTF-8'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', train=train)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, train: bool):
        super(Dataloader, self).__init__()
        self.ex = ex
        self.did = did

        if train:
            self.utt = ex['manual_transcript']
        else:
            self.utt = ex['asr_1best']
        self.utt_asr = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Dataloader.word_vocab[c] for c in self.utt]
        self.input_idx_asr = [Dataloader.word_vocab[c] for c in self.utt_asr]
        l = Dataloader.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]


class Dataloader_ver2():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Alphabet(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, train=False):
        dataset = json.load(open(data_path, 'r', encoding='UTF-8'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', train=train)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, train: bool):
        super(Dataloader_ver2, self).__init__()
        self.ex = ex
        self.did = did

        if train:
            self.utt = ex['manual_transcript']
        else:
            self.utt = ex['asr_1best']
        words = jieba.tokenize(self.utt)
        num_words = len(jieba.lcut(self.utt))
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Dataloader_ver2.word_vocab[c] for c in self.utt]
        self.input_idx_word = []
        self.idx_matrix = np.zeros((len(self.utt), num_words))
        cnt = 0
        for w in words:
            self.input_idx_word.append(Dataloader_ver2.word_vocab[w[0]])
            self.idx_matrix[w[1]: w[2], cnt] = 1
            cnt += 1

        l = Dataloader_ver2.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]