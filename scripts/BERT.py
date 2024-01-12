# coding: utf-8

import sys, os, time, gc, json
from torch.optim import Adam

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from utils.args import init_args
from utils.initialization import *
from utils.data_preprocess import SLU
from utils.vocab import PAD
from model.bert_fintune import BERT
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR

os.makedirs('trained_models', exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
# initialization params, output path, logger, random seed and torch.device
sys.argv.append('--device')
sys.argv.append('0')
sys.argv.append('--hidden_size')
sys.argv.append('768')
sys.argv.append('--lr')
sys.argv.append('1e-5')
# sys.argv.append('--testing')
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')

train_dataset = SLU(args.dataroot, train_path)
dev_dataset = SLU(args.dataroot, dev_path, test=True)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.num_tags = SLU.label_vocab.num_tags
args.tag_pad_idx = SLU.label_vocab.convert_tag_to_idx(PAD)


# 只返回tokenize以后的输入句子和对应的标签用于训练
def collote_fn_train(batch_samples):
    batch_sentence  = []
    batch_label = []
    for sample in batch_samples:
        batch_sentence.append(sample['utt'])
        batch_label.append(torch.Tensor(sample['tag_id']))
    X = tokenizer(
        batch_sentence,
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = pad_sequence(batch_label, batch_first=True, padding_value=0).type(torch.long)
    return X, y
# 除了返回输入和输出向量，还返回对应句子和slot-value对
def collote_fn_test(batch_samples):
    batch_sentence  = []
    batch_tags = []
    batch_label = []
    for sample in batch_samples:
        batch_sentence.append(sample['utt'])
        batch_tags.append(torch.Tensor(sample['tag_id']))
        batch_label.append(sample['slotvalue'])
    X = tokenizer(
        batch_sentence,
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    y = pad_sequence(batch_tags, batch_first=True, padding_value=0).type(torch.long)
    return X, y, batch_sentence, batch_label

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collote_fn_train)
train_for_val_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn_test)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn_test)

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    return optimizer, scheduler

model = BERT(args).to(device)

if args.testing:
    check_point = torch.load(open('trained_models/BERT.bin', 'rb'), map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")
    
def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_for_val_loader if choice == 'train' else dev_loader
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for current_batch in dataset:
            # current_batch[0]是BERT的输入，current_batch[1]是序列标注任务的label，
            # current_batch[2]是utt，current_batch[3]是SLU任务的标签
            pred, label, loss = model.decode(SLU.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch[2][j] for l in pred[j]]):
                    print(current_batch[2][j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        metrics = SLU.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count


def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = SLU(args.dataroot, test_path, test=True)
    dids = ['0-0', '1-0', '2-0']
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, collate_fn=collote_fn_test)
    predictions = {}
    with torch.no_grad():
        for batch in test_loader:
            pred = model.decode(SLU.label_vocab, batch)
            for pi, p in enumerate(pred[0]):
                did = dids[pi]
                predictions[did] = p
    test_json = json.load(open(test_path, 'r'))
    ptr = 0
    for ei, example in enumerate(test_json):
        for ui, utt in enumerate(example):
            utt['pred'] = [pred.split('-') for pred in predictions[f"{ei}-{ui}"]]
            ptr += 1
    json.dump(test_json, open(os.path.join(args.dataroot, 'prediction_BERT.json'), 'w',encoding='utf-8'), indent=4, ensure_ascii=False)

if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer, schduler = set_optimizer(model, args)
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        model.train()
        count = 0
        for batch in train_loader:
            # if(batch[0].input_ids.shape != batch[1].shape):
            #     continue
            output, loss = model(batch)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()
        schduler.step()
        
        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('trained_models/BERT.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))







