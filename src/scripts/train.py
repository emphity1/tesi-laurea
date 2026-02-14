"""
Training Script for MobileNetECA on CIFAR-10
Clean, modular implementation with improved progress display
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
import os
import time
from datetime import datetime
import argparse

from model import MobileNetECA, count_parameters, format_number


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Create CIFAR-10 data loaders with proper augmentation
    
    CRITICAL: Data augmentation ONLY on training set, NOT on test set
    """
    
    # Training transformations (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Test transformations (augmentation-free)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    # Use absolute path to existing dataset
    data_root = '/workspace/tesi-laurea/data'
    train_dataset = datasets.CIFAR10(root=data_root, train=True, 
                                     download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, 
                                    download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch with compact progress display"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Compact progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100. * correct / total
            progress = (batch_idx + 1) / len(loader) * 100
            
            print(f'\rEpoch {epoch}/{total_epochs} [{progress:5.1f}%] '
                  f'Loss: {avg_loss:.4f} | Acc: {acc:6.2f}%', end='')
    
    final_acc = 100. * correct / total
    final_loss = running_loss / len(loader)
    print()  # New line after epoch
    
    return final_acc, final_loss


def validate(model, loader, criterion, device):
    """Validate model on test set"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    
    return accuracy, avg_loss


def save_checkpoint(model, optimizer, epoch, train_acc, val_acc, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'val_acc': val_acc,
    }
    torch.save(checkpoint, save_path)


def save_metrics(metrics, save_path):
    """Save training metrics to JSON"""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def train_model(config):
    """
    Main training function
    
    Args:
        config: Dictionary with training configuration
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"TRAINING MOBILENETECA ON CIFAR-10")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 2)
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Model
    print("\nInitializing model...")
    model = MobileNetECA(
        num_classes=10,
        width_mult=config['width_mult'],
        lr_scale=config['lr_scale']
    ).to(device)
    
    total_params, trainable_params = count_parameters(model)
    print(f"  Parameters: {format_number(total_params)} (trainable: {format_number(trainable_params)})")
    
    # Optimizer & Scheduler
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    # Training configuration summary
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Weight decay: {config['weight_decay']}")
    print(f"  Width multiplier: {config['width_mult']}")
    print(f"  LR scale: {config['lr_scale']}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    metrics = {
        'config': config,
        'epochs': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    start_time = time.time()
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        # Train
        train_acc, train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config['epochs']
        )
        
        # Validate
        val_acc, val_loss = validate(model, test_loader, criterion, device)
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Epoch time
        epoch_time = time.time() - epoch_start
        
        # Compact summary with color-coded validation accuracy
        val_indicator = "↑" if val_acc > best_val_acc else " "
        print(f"  Val: {val_acc:6.2f}% {val_indicator} | Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_acc': round(train_acc, 2),
            'train_loss': round(train_loss, 4),
            'val_acc': round(val_acc, 2),
            'val_loss': round(val_loss, 4),
            'lr': current_lr,
            'time': round(epoch_time, 1)
        }
        metrics['epochs'].append(epoch_metrics)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            metrics['best_val_acc'] = round(best_val_acc, 2)
            metrics['best_epoch'] = epoch
            
            save_path = os.path.join(config['save_dir'], 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, train_acc, val_acc, save_path)
        
        # Save metrics periodically
        if epoch % 10 == 0 or epoch == config['epochs']:
            metrics_path = os.path.join(config['log_dir'], 'training_metrics.json')
            save_metrics(metrics, metrics_path)
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {metrics['best_epoch']})")
    print(f"Final train accuracy: {train_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'final_model.pt')
    save_checkpoint(model, optimizer, config['epochs'], train_acc, val_acc, final_path)
    
    # Save final metrics
    metrics['total_time_minutes'] = round(total_time / 60, 2)
    metrics_path = os.path.join(config['log_dir'], 'training_metrics.json')
    save_metrics(metrics, metrics_path)
    
    print(f"\nModel saved to: {config['save_dir']}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"{'='*60}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train MobileNetECA on CIFAR-10')
    
    # Model parameters
    parser.add_argument('--width_mult', type=float, default=0.42,
                       help='Width multiplier (default: 0.42)')
    parser.add_argument('--lr_scale', type=float, default=1.54,
                       help='LR scale for gradient scaling (default: 1.54)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.025,
                       help='Initial learning rate (default: 0.025)')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                       help='Weight decay (default: 3e-4)')
    
    # Infrastructure
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers (default: 2)')
    parser.add_argument('--save_dir', type=str, default='../../models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='../../reports',
                       help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Train model
    train_model(config)


if __name__ == '__main__':
    main()
