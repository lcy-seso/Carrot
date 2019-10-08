import sys

sys.path.append('../../')

import argparse
import torch

from torch import optim
from torch.nn import CrossEntropyLoss
from torchtext.data import Iterator

from model import RNNModel
from utils.data.ptb import train_dataset, valid_dataset, test_dataset, vocab

ntokens = len(vocab.itos)
criterion = CrossEntropyLoss()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch.text, batch.target
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output.view(-1, ntokens), target.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

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
            test_loss += criterion(output.view(-1, ntokens),
                                   target.view(-1)).item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-size', type=int, default=100, metavar='N', help='batch size')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        metavar='N',
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        metavar='N',
        help='upper epoch limit')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        metavar='LR',
        help='initial learning rate')
    parser.add_argument(
        '--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument(
        '--nhid',
        type=int,
        default=200,
        help='number of hidden units per layer')
    parser.add_argument(
        '--clip', type=float, default=1, help='gradient clipping')

    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='report interval')
    parser.add_argument(
        '--save',
        type=str,
        default='model.pt',
        help='path to save the final model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader, valid_loader, test_loader = Iterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        shuffle=True,
        batch_size=args.batch_size,
        device=device,
        sort_key=lambda x: len(x.text))

    model = RNNModel(ntokens, args.emsize, args.nhid).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "RNNLM.pt")


if __name__ == "__main__":
    main()
