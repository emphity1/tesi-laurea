"""
Esperimento F: Knowledge Distillation + EMA.
Student: MobileNetECA-Rep-AdvAug
Teacher: RepVGG-A0 (pre-addestrato, 94.19%)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines'))

import argparse
import math
import copy
import json
import time
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from shared_config import (
    SEED, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, MOMENTUM,
    WIDTH_MULT, NUM_CLASSES, MIN_CHANNELS, BLOCK_SETTINGS,
    set_seed, get_device, setup_logging,
    get_transforms_advanced, get_dataloaders,
    initialize_weights, count_flops
)

# Importa l'architettura dello studente
from train_E_experiments import MobileNetECARep, ModelEMA

# Importa l'architettura del teacher
from train_repvgg import RepVGG_CIFAR

# =============================================
# HYPERPARAMETERS PER KNOWLEDGE DISTILLATION
# =============================================
KD_T = 4.0      # Temperature
KD_ALPHA = 0.5  # Peso per la loss di distillazione (0.5 = 50% CE, 50% KD)
USE_EMA = True
EMA_DECAY = 0.999

# Percorso pesi del teacher
TEACHER_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'baselines', 'results_repvgg_a0', 'best_model_deploy.pth')

def kd_loss(student_logits, teacher_logits, targets, temperature, alpha):
    """Calcola la Knowledge Distillation Loss"""
    # Standard CrossEntropy loss
    ce_loss = F.cross_entropy(student_logits, targets)
    
    # KD loss (KL Divergence sulle probabilità "ammorbidite")
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=1)
    
    # KLDivLoss in PyTorch calcola: soft_targets * (log(soft_targets) - soft_prob)
    # Moltiplichiamo per T^2 per bilanciare la scala dei gradienti
    kd_l = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    return (1. - alpha) * ce_loss + alpha * kd_l

def train_and_evaluate_kd(student, teacher, run_dir, trainloader, valloader, testloader, device):
    """
    Training loop con Knowledge Distillation.
    """
    config = {
        "kd_temperature": KD_T,
        "kd_alpha": KD_ALPHA,
        "ema": USE_EMA,
        "ema_decay": EMA_DECAY,
        "base_lr": LR,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "momentum": MOMENTUM,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }

    with open(os.path.join(run_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logging.info(f"Configurazione KD: Alpha={KD_ALPHA}, Temp={KD_T}")

    optimizer = optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    if USE_EMA:
        ema = ModelEMA(student, decay=EMA_DECAY)
        logging.info(f"✓ EMA attivo: decay={EMA_DECAY}")
    else:
        ema = None

    params = sum(p.numel() for p in student.parameters())
    flops = count_flops(student)
    logging.info(f"Parametri Studente: {params:,} ({params/1000:.1f}k)")
    logging.info(f"FLOPs Studente: {flops:,} ({flops/1e6:.2f}M)")

    best_val_acc = 0.0
    stats = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_acc": [], "lr": [], "time": []
    }

    start_time = time.time()

    for epoch in range(EPOCHS):
        student.train()
        teacher.eval() # Il teacher è sempre in eval
        
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            student_logits = student(inputs)
            
            # Forward del teacher senza tracciare gradienti
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Calcolo Loss combinata KD + CE
            loss = kd_loss(student_logits, teacher_logits, targets, temperature=KD_T, alpha=KD_ALPHA)

            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5)
            optimizer.step()

            if ema is not None:
                ema.update(student)

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        avg_loss = train_loss / len(trainloader)

        # --- Validation ---
        if ema is not None:
            ema.apply_shadow(student)

        student.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = student(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), os.path.join(run_dir, "best_model.pth"))

        if ema is not None:
            ema.restore(student)

        stats["epoch"].append(epoch + 1)
        stats["train_loss"].append(avg_loss)
        stats["train_acc"].append(train_acc)
        stats["val_acc"].append(val_acc)
        stats["lr"].append(current_lr)
        stats["time"].append(time.time() - start_time)

        logging.info(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"Loss: {avg_loss:.4f} | Train: {train_acc:.2f}% | "
            f"Val: {val_acc:.2f}% (Best: {best_val_acc:.2f}%) | "
            f"LR: {current_lr:.6f}"
        )

    total_time = (time.time() - start_time) / 60
    logging.info(f"Training completato in {total_time:.1f} min")

    # --- Test Finale ---
    student.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), weights_only=True))
    student.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    test_acc = 100. * test_correct / test_total
    logging.info(f"*** Test Accuracy (finale): {test_acc:.2f}% ***")
    
    stats["test_acc_final"] = test_acc
    stats["best_val_acc"] = best_val_acc
    stats["total_params"] = params
    stats["total_flops"] = flops

    # --- Deploy ---
    student_deploy = copy.deepcopy(student)
    student_deploy.deploy()
    student_deploy.eval()
    
    deploy_correct = 0
    deploy_total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student_deploy(inputs)
            _, predicted = outputs.max(1)
            deploy_total += targets.size(0)
            deploy_correct += predicted.eq(targets).sum().item()
            
    deploy_acc = 100. * deploy_correct / deploy_total
    logging.info(f"Deploy Accuracy (verifica): {deploy_acc:.2f}%")
    
    stats["deploy_acc"] = deploy_acc
    stats["deploy_params"] = sum(p.numel() for p in student_deploy.parameters())
    
    torch.save(student_deploy.state_dict(), os.path.join(run_dir, "best_model_deploy.pth"))
    
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    set_seed()
    device = get_device()

    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_F_kd_ema")
    setup_logging(run_dir)

    logging.info("=" * 60)
    logging.info("ESPERIMENTO F: Knowledge Distillation + EMA")
    logging.info("=" * 60)

    # 1. Carica il Teacher (RepVGG-A0 in mode Deploy)
    logging.info("Caricamento Teacher: RepVGG-A0")
    teacher = RepVGG_CIFAR(num_classes=NUM_CLASSES, deploy=True)
    teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=device, weights_only=True))
    teacher = teacher.to(device)
    teacher.eval() # Importante: no variazioni per bn/dropout

    # Optional: verifica rapida accuracy del teacher
    # logging.info("Verifica Teacher Accuracy in corso...")

    # 2. Inizializza lo Studente
    logging.info("Inizializzazione Student: MobileNetECA-Rep")
    student = MobileNetECARep(num_classes=NUM_CLASSES).to(device)

    # 3. Dataloaders
    transform_train, transform_test = get_transforms_advanced()
    trainloader, valloader, testloader = get_dataloaders(transform_train, transform_test)

    # 4. Vai col training
    train_and_evaluate_kd(student, teacher, run_dir, trainloader, valloader, testloader, device)

