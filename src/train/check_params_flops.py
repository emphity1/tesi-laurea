import json
import os
import torch

from train_A_baseline import MobileNetBaseline
from train_B_eca import MobileNetECA
from train_C_eca_rep import MobileNetECARep as MobileNetECARep_C
from train_D_eca_rep_advaug import MobileNetECARep as MobileNetECARep_D
from train_E_experiments import MobileNetECARep as MobileNetECARep_E
from shared_config import count_flops

# Imposta la stessa configurazione di base usata negli script
# NUM_CLASSES = 10 (dal dataset cifar-10)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_model(model_class, model_name):
    # Istanzia il modello
    model = model_class(num_classes=10)
    
    # Parametri in modalità training
    train_params = count_parameters(model)
    
    # FLOPs in modalità training
    train_flops = count_flops(model)
    
    # Switch to deploy (se supportato)
    deploy_params = None
    deploy_flops = None
    if hasattr(model, 'deploy'):
        model.deploy()
        deploy_params = count_parameters(model)
        deploy_flops = count_flops(model)
        
    print(f"--- {model_name} ---")
    print(f"Train Params:  {train_params:,}")
    if deploy_params is not None:
        print(f"Deploy Params: {deploy_params:,} (Diff: {deploy_params - train_params:,})")
    print(f"Train FLOPs:   {train_flops:,}")
    if deploy_flops is not None:
        print(f"Deploy FLOPs:  {deploy_flops:,} (Diff: {deploy_flops - train_flops:,})")
    print("")

if __name__ == "__main__":
    print("ANALISI PARAMETRI E FLOPS DEI MODELLI\n")
    evaluate_model(MobileNetBaseline, "A: MobileNetV2 (Baseline)")
    evaluate_model(MobileNetECA, "B: MobileNetECA")
    evaluate_model(MobileNetECARep_C, "C: MobileNetECA-Rep")
    evaluate_model(MobileNetECARep_D, "D: MobileNetECA-Rep-AdvAug")
    evaluate_model(MobileNetECARep_E, "E: MobileNetECA-Rep-AdvAug (Experiments/KD)")
