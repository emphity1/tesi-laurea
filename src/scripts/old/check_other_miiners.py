import requests
from collections import Counter

def fetch_miner_details(base_url: str, miner_hotkey: str):
    """
    Effettua una GET su /nodes/?detailed=true e
    restituisce i dati del miner che ha l’hotkey richiesto.
    """
    url = f"{base_url}/nodes/?detailed=true"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Errore nella richiesta: {e}")
        return None
    
    data = response.json()
    return data.get(miner_hotkey)

def main():
    BASE_URL = "https://api.chutes.ai"
    HOTKEY = "5CaqyHE9eBPyN469MNKor8R3zoyNsQwCzMZjd51xAR66S8tF"

    miner_info = fetch_miner_details(BASE_URL, HOTKEY)
    if not miner_info:
        print(f"Nessun miner trovato con hotkey={HOTKEY}.")
        return
    
    # miner_info dovrebbe contenere qualcosa tipo:
    # {
    #   "provisioned": [
    #       {"gpu": "h200", "chute": {...}},
    #       {"gpu": "3090", "chute": {...}},
    #       ...
    #   ]
    # }

    provisioned = miner_info.get("provisioned", [])
    # Estraggo la lista di GPU
    gpu_list = [p.get("gpu", "unknown") for p in provisioned]

    # Uso collections.Counter per contare le occorrenze di ciascuna GPU
    gpu_counts = Counter(gpu_list)

    print(f"Conteggio GPU per miner {HOTKEY}:")
    for gpu_model, count in gpu_counts.items():
        print(f" - GPU={gpu_model}, occorrenze={count}")

if __name__ == "__main__":
    main()
