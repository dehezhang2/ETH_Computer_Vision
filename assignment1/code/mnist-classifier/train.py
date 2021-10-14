import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from lib.dataset import MNISTDataset
from lib.networks import MLPClassifier, ConvClassifier
from lib.utils import UpdatingMean


BATCH_SIZE = 16
NUM_WORKERS = 4
NUM_EPOCHS = 5


def run_training_epoch(net, optimizer, dataloader):
    loss_aggregator = UpdatingMean()
    # Put the network in training mode.
    net.train()
    # Loop over batches.
    for batch in tqdm(dataloader):
        raise NotImplementedError()
        # Reset gradients.
        # TODO

        # Forward pass.
        output = None

        # Compute the loss - cross entropy.
        # Documentation https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html.
        loss = None

        # Backwards pass.
        # TODO

        # Save loss value in the aggregator.
        loss_aggregator.add(loss.item())
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    return torch.mean((torch.argmax(output, dim=1) == labels).float())


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
    train_dataset = MNISTDataset(split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )
    
    # Create the validation dataset and dataloader.
    valid_dataset = MNISTDataset(split='test')
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    # Create the network.
    # net = MLPClassifier()
    net = ConvClassifier()

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
