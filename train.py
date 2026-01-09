import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

from model.model import Encoder

# Training configuration
CONFIG = {
    'dataset': 'cifar100',  # 'cifar10' or 'cifar100'
    'batch_size': 128,
    'epochs': 200,
    'learning_rate': 0.001,
    'weight_decay': 5e-4,
    'label_smoothing': 0.1,
    'early_stopping_patience': 20,
    'val_split': 0.1,  # fraction of training data for validation
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 2,
    'save_dir': './outputs'
}

# Model configuration (maps to Encoder arguments)
MODEL_CONFIG = {
    'n_blocks': 4,
    'kernel_sizes': [5, 5, 3, 3],
    'channels': [3, 32, 64, 128, 256],  # n_blocks + 1 values (in -> out for each block)
    'paddings': [2, 2, 1, 1],
    'ratios': [4, 4, 4, 4],  # channel attention reduction ratios
}


def get_dataloaders(config):
    """Create train, validation, and test dataloaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    if config['dataset'] == 'cifar10':
        full_train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train
        )
        val_dataset_base = datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_test
        )
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test
        )
        num_classes = 10
    else:
        full_train_dataset = datasets.CIFAR100(
            root='./data', train=True, download=False, transform=transform_train
        )
        val_dataset_base = datasets.CIFAR100(
            root='./data', train=True, download=False, transform=transform_test
        )
        test_dataset = datasets.CIFAR100(
            root='./data', train=False, download=False, transform=transform_test
        )
        num_classes = 100

    # Split training data into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * config['val_split'])
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(total_size), [train_size, val_size], generator=generator
    )

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices.indices)
    val_dataset = torch.utils.data.Subset(val_dataset_base, val_indices.indices)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers']
    )

    return train_loader, val_loader, test_loader, num_classes


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def plot_metrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, save_dir):
    """Plot and save training metrics."""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'g-', label='Val Loss')
    axes[0].plot(epochs, test_losses, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training, Validation, and Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-', label='Train Accuracy')
    axes[1].plot(epochs, val_accs, 'g-', label='Val Accuracy')
    axes[1].plot(epochs, test_accs, 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training, Validation, and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150)
    plt.close()
    print(f"Saved metrics plot to {save_dir}/training_metrics.png")


def main():
    config = CONFIG
    device = config['device']
    print(f"Using device: {device}")

    os.makedirs(config['save_dir'], exist_ok=True)

    # Data
    print("Loading data...")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(config)
    print(f"Dataset: {config['dataset']} | Classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Val samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # Model
    model = Encoder(
        n_blocks=MODEL_CONFIG['n_blocks'],
        kernel_sizes=MODEL_CONFIG['kernel_sizes'],
        channels=MODEL_CONFIG['channels'],
        paddings=MODEL_CONFIG['paddings'],
        ratios=MODEL_CONFIG['ratios'],
        n_classes=num_classes
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Tracking
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []
    best_val_acc = 0.0
    patience_counter = 0

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch:3d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}%")

        # Save best model and check early stopping (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'best_test_acc': best_test_acc,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience_counter} epochs)")
                break

    # Final plots
    plot_metrics(train_losses, val_losses, test_losses, train_accs, val_accs, test_accs, config['save_dir'])

    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.2f}% | Test accuracy: {best_test_acc:.2f}%")
    print(f"Model saved to {config['save_dir']}/best_model.pth")


if __name__ == '__main__':
    main()
