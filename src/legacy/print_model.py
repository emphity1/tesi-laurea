import torch
import torch.nn as nn
import os
import json
import datetime
import sys

# Percorso del modello salvato
model_path = '/workspace/Dima/models/luaqi.pt'
log_file_path = '/workspace/logs.txt'

activations = {}
pre_backprop_grads = {}
post_backprop_grads = {}

def load_model(model_path):
    model = None
    error_messages = {}

    # Prova a caricare con torch.load
    try:
        model = torch.load(model_path)
        log_message("Modello caricato con torch.load")
    except Exception as e:
        error_messages["torch.load"] = str(e)

    # Se torch.load fallisce, prova con torch.jit.load
    if model is None:
        try:
            model = torch.jit.load(model_path)
            log_message("Modello caricato con torch.jit.load")
        except Exception as e:
            error_messages["torch.jit.load"] = str(e)

    # Prova a caricare con map_location per renderlo compatibile con CPU
    if model is None:
        try:
            model = torch.load(model_path, map_location=torch.device('cpu'))
            log_message("Modello caricato con torch.load e map_location")
        except Exception as e:
            error_messages["torch.load with map_location"] = str(e)

    # Se nessuno dei metodi ha funzionato, stampa gli errori
    if model is None:
        log_message("Nessuno dei metodi di caricamento ha funzionato. Ecco gli errori riscontrati:")
        for method, error in error_messages.items():
            log_message(f"{method}: {error}")
    return model

def log_message(message):
    """Funzione per registrare i messaggi nel file di log con timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def get_activation(name):
    """Hook per monitorare le attivazioni."""
    def hook(model, input, output):
        activations[name] = output.detach()
        log_message(f"Attivazione layer {name}: media={output.mean().item()}, varianza={output.var().item()}, "
                    f"min={output.min().item()}, max={output.max().item()}")
    return hook

def register_hooks(model):
    """Registra gli hook per i layer di attivazione, inclusi ReLU, Sigmoid, Tanh e GELU."""
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.GELU)):  # Aggiunto GELU
            layer.register_forward_hook(get_activation(name))

def store_gradients_pre_backprop(model):
    """Conserva i gradienti prima del backward pass."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            pre_backprop_grads[name] = param.grad.clone() if param.grad is not None else None
            if param.grad is not None:
                log_message(f"Gradiente pre-backprop per {name}: media={param.grad.mean().item()}, varianza={param.grad.var().item()}")

def store_gradients_post_backprop(model):
    """Conserva i gradienti dopo il backward pass."""
    for name, param in model.named_parameters():
        if param.requires_grad:
            post_backprop_grads[name] = param.grad.clone() if param.grad is not None else None
            if param.grad is not None:
                log_message(f"Gradiente post-backprop per {name}: media={param.grad.mean().item()}, varianza={param.grad.var().item()}")

def compare_gradients():
    """Confronta i gradienti pre e post backpropagation."""
    for name in pre_backprop_grads:
        if pre_backprop_grads[name] is not None and post_backprop_grads[name] is not None:
            diff = (post_backprop_grads[name] - pre_backprop_grads[name]).abs().mean().item()
            log_message(f"Diff tra gradiente pre e post-backprop per {name}: media della differenza={diff}")

def explore_model(model):
    if model:
        log_message("=== Inizio esplorazione del modello ===\n")

        # Forza il modello sulla CPU
        model = model.to(torch.device('cpu'))

        # Informazioni generali sul modello
        log_message("Informazioni generali sul modello:")
        log_message(f"Tipo di modello: {type(model)}")
        log_message(f"Numero totale di parametri: {count_parameters(model)}")
        log_message(f"Numero di parametri addestrabili: {count_trainable_parameters(model)}\n")

        # Stampa la struttura del modello
        log_message("Struttura del modello:")
        log_message(str(model) + "\n")

        # Registrare gli hook per monitorare le attivazioni
        register_hooks(model)

        # Esegui una forward pass fittizia con dei dati di esempio
        input_data = torch.randn(1, 3, 224, 224).to(torch.device('cpu'))  # Assicurati che gli input siano sulla CPU
        output = model(input_data)

        # Monitoraggio gradienti
        log_message("Monitoraggio dei gradienti:")
        store_gradients_pre_backprop(model)
        output.mean().backward()  # Simula il backward pass per calcolare i gradienti
        store_gradients_post_backprop(model)
        compare_gradients()





        state_dict = model.state_dict()
        for key, value in state_dict.items():
            log_message(f"Chiave: {key}, Forma: {value.shape}, Tipo: {value.dtype}")

        # Verificare se ci sono hook registrati nel modello
        log_message("Controllo per eventuali hook registrati:")
        for name, module in model.named_modules():
            if module._backward_hooks or module._forward_hooks:
                log_message(f"Layer con hook registrati: {name}, Tipo: {type(module)}")
                log_message(f"  - Hook di backprop: {module._backward_hooks}")
                log_message(f"  - Hook di forward: {module._forward_hooks}")

            # Controllare se ci sono metadati nel modello
        log_message("Ricerca di metadati di training:")
        if hasattr(model, 'metadata'):
            metadata = model.metadata
            try:
                metadata_str = json.dumps(metadata, indent=4)
                log_message(f"Metadati trovati: {metadata_str}")
            except TypeError:
                log_message(f"Metadati trovati: {metadata}")
        else:
            log_message("Nessun metadato relativo al training trovato.")

        log_message("=== Fine esplorazione del modello ===\n")
    else:
        log_message("Il modello non è stato caricato correttamente.")


def count_parameters(model):
    """Conta il numero totale di parametri nel modello."""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    """Conta il numero di parametri addestrabili nel modello."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_environment():
    """Logga informazioni sull'ambiente di esecuzione."""
    log_message("=== Informazioni sull'ambiente ===")
    log_message(f"Python versione: {sys.version}")
    log_message(f"PyTorch versione: {torch.__version__}")
    log_message(f"Sistema operativo: {os.uname().sysname} {os.uname().release} {os.uname().version}\n")
    log_message("=== Fine informazioni sull'ambiente ===\n")

def main():
    # Creare la cartella workspace se non esiste
    os.makedirs('/workspace', exist_ok=True)

    # Logga l'inizio del processo con timestamp
    log_message("Inizio processo di caricamento ed esplorazione del modello.")

    # Logga informazioni sull'ambiente
    log_environment()

    # Carica il modello
    model = load_model(model_path)

    # Esplora il modello
    explore_model(model)

    # Logga la fine del processo
    log_message("Fine processo di caricamento ed esplorazione del modello.")

    print(f"L'esplorazione del modello è stata salvata in {log_file_path}")

if __name__ == "__main__":
    main()
