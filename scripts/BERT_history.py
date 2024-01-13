# coding: utf-8

import torch
import sys, os, time, json
install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)
from model.decoder import NomalDecoder, HistoryDecoder
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils.args import init_args
from utils.evaluator import Evaluator_history
from utils.initialization import *
from utils.Semantic_triplet import get_triplet
from dataset.dataset import LabelVocab_history, get_dataset

os.makedirs('trained_models', exist_ok=True)

# initialization params, output path, logger, random seed and torch.device
sys.argv.append('--hidden_size')
sys.argv.append('128')
sys.argv.append('--num_layer')
sys.argv.append('1')
sys.argv.append('--encoder_cell')
sys.argv.append('GRU')
# sys.argv.append('--lr')
# sys.argv.append('1e-5')
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

# get dataset
label_converter = LabelVocab_history('data/ontology.json')
dataset_dir = 'processed_data_v2'
os.makedirs(dataset_dir, exist_ok=True)
train_dataset = get_dataset('data/train.json', label_converter, os.path.join(dataset_dir, 'train_data'), args)
dev_dataset = get_dataset('data/development.json', label_converter, os.path.join(dataset_dir, 'dev_data'), args)
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

def collate_func(batch):
    return tuple(zip(*batch))

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
dev_data_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=collate_func)
encoding_len = train_dataset[0][0][0].vector_with_noise.shape[1]

# model configuration
decoder = HistoryDecoder(encoding_len, label_converter.num_tags, args).to(device)
optimizer = Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.CrossEntropyLoss()

print('Start training ......')
print('Total training epochs: %d' % (args.max_epoch))
best_acc = 0
for epoch in range(args.max_epoch):
    total_loss = 0
    # training
    start_time = time.time()
    decoder.train()
    for batch_x, batch_y in train_data_loader:
        batch_loss = 0
        for round_x, round_y in zip(batch_x, batch_y):
            decoder.reset()
            for x, y in zip(round_x, round_y):
                output = decoder(x.vector_without_noise)
                # output = decoder(x.vector_with_noise)
                loss = loss_fn(output, y)
                total_loss += loss
                batch_loss += loss
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss.item() / len(train_dataset)
    print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (epoch, time.time() - start_time, avg_loss))

    # validation
    start_time = time.time()
    evaluator = Evaluator_history()
    decoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dev_data_loader:
            for round_x, round_y in zip(batch_x, batch_y):
                decoder.reset()
                for x, y in zip(round_x, round_y):
                    input_vector = x.vector_with_noise if args.noise else x.vector_without_noise
                    output = decoder(input_vector)
                    loss = loss_fn(output, y)
                    total_loss += loss
                    input_tokens = x.tokens_with_noise if args.noise else x.tokens_without_noise
                    prediction = get_triplet(input_tokens, output, label_converter)
                    expected = get_triplet(x.tokens_with_noise, y, label_converter)
                    evaluator.add_result(prediction, expected)
    acc = evaluator.accuracy_rate
    precision, recall, f1_score = evaluator.precision_rate, evaluator.recall_rate, evaluator.f1_score
    avg_loss = total_loss.item() / len(dev_dataset)
    print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.3f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' %
          (epoch, time.time() - start_time, acc, precision, recall, f1_score))

    if acc > best_acc:
        best_acc = acc
        best_f1_score = f1_score
        best_precision = precision
        best_recall = recall
        torch.save({
            'epoch': epoch,
            'model': decoder.state_dict(),
            'optim': optimizer.state_dict(),
            'seed': args.seed
        }, f'trained_models/slu-bert-history-{args.encoder_cell}.bin')

