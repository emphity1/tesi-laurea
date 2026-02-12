# Capitolo 1: Introduzione

> **Obiettivo**: Presentare il problema, motivare la tesi, e definire gli obiettivi.  
> **Lunghezza**: 5-6 pagine

---

## 1.1 Contesto e Motivazione (~2 pagine)

### Cosa scrivere:
- **Esplosione dell'AI su dispositivi edge**: smartphone, IoT, droni, wearables
- **Problema principale**: modelli potenti (ResNet-50, VGG) richiedono troppe risorse
  - Memoria: centinaia di MB
  - Computazione: miliardi di operazioni
  - Energia: batterie si scaricano rapidamente
- **Necessità di efficienza**: Edge AI richiede modelli compatti ma accurati
- **CIFAR-10 come benchmark**: dataset standard per testare architetture efficienti

### Template paragrafo:

```
L'intelligenza artificiale ha visto una crescita esponenziale negli ultimi anni, 
con applicazioni che spaziano dal riconoscimento vocale alla visione artificiale. 
Tuttavia, l'implementazione di modelli di deep learning su dispositivi edge 
(smartphone, dispositivi IoT, droni) presenta sfide significative legate alle 
risorse limitate. I modelli tradizionali come ResNet-50 [He et al., 2016] o 
VGG-16 [Simonyan & Zisserman, 2014] richiedono centinaia di megabyte di memoria 
e miliardi di operazioni floating-point (FLOPs), rendendoli inadatti per 
hardware con vincoli energetici e computazionali stringenti.

Il dataset CIFAR-10 [Krizhevsky, 2009] rappresenta un benchmark fondamentale 
per lo sviluppo di architetture efficienti. Composto da 60.000 immagini RGB 
di dimensione 32×32 pixel suddivise in 10 classi, CIFAR-10 offre una sfida 
intermedia tra dataset troppo semplici (MNIST) e troppo complessi (ImageNet). 
La sua dimensione ridotta lo rende ideale per sperimentare rapidamente nuove 
architetture mantenendo una complessità sufficiente a validarne l'efficacia.
```

### Punti chiave da menzionare:
- [ ] Crescita dell'AI su edge devices
- [ ] Vincoli di memoria e computazione
- [ ] Consumo energetico come fattore critico
- [ ] CIFAR-10 come benchmark standard
- [ ] Trade-off accuratezza vs efficienza

---

## 1.2 Obiettivi della Tesi (~2 pagine)

### Cosa scrivere:
- **Obiettivo principale**: Sviluppare un modello ultra-compatto per CIFAR-10
  - Target: < 250k parametri
  - Target: > 85% accuracy
  - Target: < 50M FLOPs
- **Obiettivi secondari**:
  - Confrontare con architetture state-of-the-art
  - Dimostrare l'efficacia di ECA attention
  - Validare l'uso di GELU su dataset compatti

### Template paragrafo:

```
Gli obiettivi della presente tesi sono i seguenti:

1. **Sviluppo di un'architettura efficiente**: Progettare e implementare una 
   rete neurale convoluzionale con meno di 250.000 parametri addestrabili, 
   mantenendo un'accuratezza superiore all'85% sul dataset CIFAR-10.

2. **Integrazione di meccanismi di attenzione leggeri**: Incorporare il 
   meccanismo ECA (Efficient Channel Attention) [Wang et al., 2020] per 
   migliorare la capacità rappresentativa del modello senza incrementare 
   significativamente la complessità computazionale.

3. **Analisi comparativa**: Confrontare le prestazioni del modello proposto 
   con architetture efficienti esistenti (MobileNetV2, ShuffleNetV2, ResNet-20) 
   in termini di accuratezza, numero di parametri e complessità computazionale.

4. **Validazione dell'approccio**: Dimostrare che è possibile ottenere un 
   buon compromesso tra efficienza e prestazioni attraverso scelte di design 
   mirate (ECA attention, GELU activation, width multiplier ottimizzato).
```

### Tabella obiettivi quantitativi:

| Metrica | Target | Motivazione |
|---------|--------|-------------|
| Parametri | < 250k | Constraint memoria per MCU |
| Accuratezza | > 85% | Soglia usabilità pratica |
| FLOPs | < 50M | Limitazione computazionale |
| Dimensione modello | < 1MB | Storage limitato su edge |

---

## 1.3 Contributi della Tesi (~1 pagina)

### Cosa scrivere:
- **Architettura Mimir**: Combinazione originale di componenti esistenti
- **Analisi comparativa**: Dati empirici su trade-off
- **Dimostrazione pratica**: Fattibilità di edge AI con risorse minime

### Template paragrafo:

```
I principali contributi di questa tesi includono:

- **Architettura Mimir**: Un modello compatto (~200k parametri) che combina 
  Inverted Residual blocks, ECA attention e GELU activation per ottenere 
  88% di accuratezza su CIFAR-10.

- **Analisi empirica del trade-off efficienza-accuratezza**: Confronto 
  quantitativo con tre architetture baseline, evidenziando i vantaggi 
  dell'approccio proposto in termini di rapporto parametri/prestazioni.

- **Validazione di design choices**: Dimostrazione dell'efficacia di ECA 
  rispetto a SE e di GELU rispetto a ReLU in contesti resource-constrained.
```

---

## 1.4 Struttura della Tesi (~0.5 pagine)

### Template paragrafo:

```
La tesi è organizzata come segue:

- **Capitolo 2** presenta lo stato dell'arte, analizzando le architetture 
  efficienti esistenti (MobileNet, ShuffleNet, ResNet) e i meccanismi di 
  attenzione (SE, ECA).

- **Capitolo 3** descrive la metodologia, dettagliando l'architettura Mimir, 
  le scelte di design e il setup sperimentale.

- **Capitolo 4** riporta i risultati sperimentali, confrontando il modello 
  proposto con le baseline in termini di accuratezza, parametri e FLOPs.

- **Capitolo 5** conclude la tesi, discutendo limitazioni e possibili 
  sviluppi futuri.
```

---

## Checklist Capitolo 1

Prima di considerare completo il capitolo, verifica:

- [ ] Motivazione chiara del problema (edge AI)
- [ ] CIFAR-10 presentato come benchmark appropriato
- [ ] Obiettivi quantitativi ben definiti (< 250k params, > 85% acc)
- [ ] Contributi della tesi esplicitati
- [ ] Struttura dei capitoli descritta
- [ ] Riferimenti bibliografici citati correttamente
- [ ] Lunghezza: 5-6 pagine

---

## Suggerimenti di scrittura

1. **Tono**: Professionale ma accessibile. Evita gergo inutile.
2. **Tempo verbale**: Presente per fatti generali, passato per il lavoro svolto
3. **Evita**: "Io ho fatto", "La mia tesi". Usa "Si è sviluppato", "Questo lavoro presenta"
4. **Acronimi**: Sempre spiegati alla prima occorrenza
5. **Figure**: Se includi grafici, citali nel testo ("Come mostrato in Figura 1.1...")
