#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar


def test(net_g, data_loader, device):
    """
    Evaluate the model performance on the test dataset.

    Args:
        net_g: The global model to evaluate.
        data_loader: DataLoader for the test dataset.
        device: The device on which to perform the computation.
    """
    net_g.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            log_probs = net_g(data)
            test_loss += F.cross_entropy(log_probs, target).item()
            y_pred = log_probs.argmax(dim=1, keepdim=True)
            correct += y_pred.eq(target.view_as(y_pred)).sum().item()

            # Collect predictions and targets for F1-Score calculation
            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    test_loss /= len(data_loader.dataset)
    accuracy = 100. * correct / len(data_loader.dataset)
    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%), F1-Score: {f1:.4f}\n')
    return accuracy, test_loss, f1


def train(net_g, train_loader, optimizer, device, writer, epoch):
    """
    Train the model for one epoch.

    Args:
        net_g: The model to train.
        train_loader: DataLoader for the training dataset.
        optimizer: Optimizer for model parameters.
        device: The device on which to perform the computation.
        writer: TensorBoard writer for logging metrics.
        epoch: Current epoch number.
    """
    net_g.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net_g(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    print(f'\nEpoch {epoch}: Average training loss: {avg_loss:.4f}')
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    return avg_loss


if __name__ == '__main__':
    # Parse arguments
    args = args_parser()
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    torch.manual_seed(args.seed)

    # Load dataset
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=transform)
    elif args.dataset == 'cifar':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar/', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data/cifar/', train=False, download=True, transform=transform)
    else:
        raise ValueError('Unrecognized dataset')

    # Prepare data loaders
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)

    # Initialize model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(device)
        elif args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(device)
    elif args.model == 'mlp':
        input_dim = int(torch.prod(torch.tensor(dataset_train[0][0].shape)))
        net_glob = MLP(dim_in=input_dim, dim_hidden=64, dim_out=args.num_classes).to(device)
    else:
        raise ValueError('Unrecognized model type')

    print(net_glob)

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Reduce LR every 10 epochs by a factor of 0.5

    # TensorBoard setup
    writer = SummaryWriter(log_dir='./logs')

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(net_glob, train_loader, optimizer, device, writer, epoch)
        scheduler.step()  # Adjust learning rate

    # Save training loss plot
    writer.close()

    # Testing
    print(f'\nTesting on {len(dataset_test)} samples...')
    test_acc, test_loss, test_f1 = test(net_glob, test_loader, device)
