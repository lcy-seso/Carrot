import sys

sys.path.append('../../')

import argparse
from collections import defaultdict
from time import time
from datetime import datetime
import torch

from torch.optim import Adam
from torch.nn import LSTM
from torch.nn.functional import cross_entropy
from torchtext.data import Iterator

from model.LanguageModel import LanguageModel
from model.loop_visible_lstm import DefaultCellLSTM, JITDefaultCellLSTM, \
    FineGrainedCellLSTM, JITFineGrainedCellLSTM

from utils.data.ptb import train_dataset, valid_dataset, test_dataset, \
    vocab_size


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch.text, batch.target
        optimizer.zero_grad()
        output, _ = model(data)
        loss = cross_entropy(output.view(-1, vocab_size), target.view(-1))
        loss.backward()

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch.text, batch.target
            output, _ = model(data)
            test_loss += cross_entropy(output.view(-1, vocab_size),
                                       target.view(-1)).item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lstm', type=str,
                        choices=['LSTM', 'DefaultCellLSTM',
                                 'JITDefaultCellLSTM',
                                 'FineGrainedCellLSTM',
                                 'JITFineGrainedCellLSTM'])
    parser.add_argument('--num-layers', type=int, required=True)
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--jit', action='store_true')

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)

    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--log-interval', type=int, default=100)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    train_loader, valid_loader, test_loader = Iterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        shuffle=False,
        batch_size=args.batch_size,
        device=device,
        sort_key=lambda x: len(x.text))

    record = defaultdict(dict)

    def run(jit, lstm):
        lstm_name = lstm.__name__
        print('{} jit: {}'.format(lstm_name, jit))

        if jit and lstm == LSTM:
            print("`torch.nn.LSTM` can't run with `jit` flag `True`")
            record[jit][lstm_name] = 0
            return

        start = time()
        model = LanguageModel(jit, lstm, vocab_size, args.embedding_size,
                              args.hidden_size, args.num_layers,
                              args.bidirectional).to(device)

        optimizer = Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epoch + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(args, model, device, test_loader)
        end = time()
        record[jit][lstm_name] = end - start
        print('------')

    if args.lstm is None:
        LSTM_LIST = [LSTM, JITDefaultCellLSTM, DefaultCellLSTM,
                     JITFineGrainedCellLSTM, FineGrainedCellLSTM]
        for lstm in LSTM_LIST:
            for jit in [True, False]:
                run(jit, lstm)

        LSTM_NAME_LIST = list(map(lambda x: x.__name__, LSTM_LIST))

        with open(datetime.now().strftime("%H:%M:%S"), "w") as f:
            f.write(' {}'.format('    '.join(LSTM_NAME_LIST)))
            f.write('\n')

            f.write('no-jit	{}'.format(
                '    '.join(
                    map(lambda x: str(record[False][x]), LSTM_NAME_LIST))))
            f.write('\n')

            f.write('jit	{}'.format(
                '    '.join(
                    map(lambda x: str(record[True][x]), LSTM_NAME_LIST))))
    else:
        if args.lstm == 'LSTM':
            lstm = LSTM
        elif args.lstm == 'DefaultCellLSTM':
            lstm = DefaultCellLSTM
        elif args.lstm == 'JITDefaultCellLSTM':
            lstm = JITDefaultCellLSTM
        elif args.lstm == 'FineGrainedCellLSTM':
            lstm = FineGrainedCellLSTM
        elif args.lstm == 'JITFineGrainedCellLSTM':
            lstm = JITFineGrainedCellLSTM
        else:
            raise ValueError("Unsupported LSTM type %s" % args.lstm)
        run(args.jit, lstm)


if __name__ == "__main__":
    main()
