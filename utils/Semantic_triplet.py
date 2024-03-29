from typing import List, Tuple

import torch
from utils.vocab import Label, LabelVocab


def get_triplet(text: List[str], output: torch.Tensor, label_converter: LabelVocab) -> List[Tuple[str, str, str]]:
    ret = []
    output = output[1:-1].argmax(dim=1)
    labels = [label_converter.convert_idx_to_tag(i.item()) for i in output]
    labels.append(Label(2, '', ''))
    start = -1
    act = ''
    slot = ''
    for i, v in enumerate(labels):
        if v.bio == 0:
            if start != -1:
                value = ''.join(text[start:i])
                ret.append([act, slot, value])
                start = -1
            start = i
            act = v.act
            slot = v.slot
        elif v.bio == 2 and start != -1:
            value = ''.join(text[start:i])
            ret.append([act, slot, value])
            start = -1
        elif v.bio == 1 and (v.act, v.slot) != (act, slot):
            # invalid tag sequence
            return []
    return ret