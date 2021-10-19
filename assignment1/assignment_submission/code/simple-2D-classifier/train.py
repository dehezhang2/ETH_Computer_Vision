import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from lib.dataset import Simple2DDataset, Simple2DTransformDataset
from lib.networks import LinearClassifier, MLPClassifier
from lib.utils import UpdatingMean


BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 10


def run_training_epoch(net, optimizer, dataloader):
    loss_aggregator = UpdatingMean()
    # Put the network in training mode.
    net.train()
    # Loop over batches.
    for batch in dataloader:
        # raise NotImplementedError()
        # Reset gradients.
        optimizer.zero_grad()
        # Forward pass.
        output = net(batch['input'])

        # Compute the loss - binary cross entropy.
        # Documentation https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy.html.
        loss = F.binary_cross_entropy(output, batch['annotation'])

        # Backwards pass.
        loss.backward()
        optimizer.step()

        # Save loss value in the aggregator.
        loss_aggregator.add(loss.item())
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    return torch.mean(((output >= 0.5) == labels).float())


def run_validation_epoch(net, dataloader):
    accuracy_aggregator = UpdatingMean()
    # Put the network in evaluation mode.
    net.eval()
    # Loop over batches.
    for batch in dataloader:
        # Forward pass only.
        output = net(batch['input'])
    
        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch['annotation'])

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
    return accuracy_aggregator.mean()


if __name__ == '__main__':
    # Create the training dataset and dataloader.
    # train_dataset = Simple2DDataset(split='train')
    train_dataset = Simple2DTransformDataset(split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Create the validation dataset and dataloader.
    # valid_dataset = Simple2DDataset(split='valid')
    valid_dataset = Simple2DTransformDataset(split='valid')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    net = LinearClassifier()
    # net = MLPClassifier()

    # Create the optimizer.
    optimizer = Adam(net.parameters())

    # Main training loop.
    best_accuracy = 0
    for epoch_idx in range(NUM_EPOCHS):
        # Training code.
        loss = run_training_epoch(net, optimizer, train_dataloader)
        print('[Epoch %02d] Loss: %.4f' % (epoch_idx + 1, loss))

        # Validation code.
        acc = run_validation_epoch(net, valid_dataloader)
        print('[Epoch %02d] Acc.: %.4f' % (epoch_idx + 1, acc * 100) + '%')

        # Save checkpoint if accuracy is the best so far.
        checkpoint = {
            'epoch_idx': epoch_idx,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(checkpoint, f'best-{net.codename}.pth')
