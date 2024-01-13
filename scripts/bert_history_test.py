import sys, json, os
from dataset.dataset import LabelVocab, get_dataset
from model.decoder import NomalDecoder, HistoryDecoder

from utils.args import init_args
from utils.initialization import *
from utils.Semantic_triplet import get_triplet
from transformers import BertModel, BertTokenizer
# from transformers import AutoTokenizer, AutoModel

# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")


test_data_path = 'data/test_unlabelled.json'
output_path = 'data/prediction_BERT_history.json'

with open(test_data_path, encoding='utf-8') as f:
    test_data = json.load(f)

pretrained_bert = 'bert-base-chinese'
# pretrained_bert = 'hfl/chinese-bert-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(pretrained_bert)
bert = BertModel.from_pretrained(pretrained_bert)
label_converter = LabelVocab('data/ontology.json')

decoder = HistoryDecoder(args.embed_size, label_converter.num_tags, args).to(device)
check_point = torch.load(open('trained_models/slu-bert-history-GRU.bin', 'rb'), map_location=device)
decoder.load_state_dict(check_point['model'])

for i in test_data:
    for utt in i:
        text = utt['asr_1best']
        model_input = tokenizer(text, return_tensors='pt')
        output = bert(**model_input)[0][0].to(device).detach()
        output = decoder(output)
        pred = get_triplet(tokenizer.tokenize(text), output, label_converter)
        utt['pred'] = pred
    decoder.reset()

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)