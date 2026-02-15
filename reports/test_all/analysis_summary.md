## Risultati Sperimentali Finali (Ablation Study Completo)

## Risultati Sperimentali: Ablation Study Completo

## Risultati Sperimentali: Ablation Study Completo

| Configurazione | ECA | Attivazione | Reparam | Augmentation | Top-1 Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV2 (Golden Pure)** | ❌ | ReLU6 | ❌ | Standard | **90.87%** |
| **Baseline (Vanilla)** | ❌ | ReLU | ❌ | Standard | 89.92% |
| **MobileNetECA (Standard)** | ✅ | GELU | ❌ | Standard | **92.12%** |
| **MobileNetECA (Reparam)** | ✅ | GELU | ✅ | Standard | **92.47%** |
| **Advanced Strategy** | ✅ | GELU | ✅ | **AutoAug + Cutout** | **93.50%** |

## Analisi dei Contributi Singoli
Dallo studio delle combinazioni emerge chiaramente il ruolo di ogni componente:
1.  **Architettura (+1.60%)**: L'ottimizzazione strutturale (ECA + GELU + Reparam) ha portato il modello dal 90.87% al 92.47%.
2.  **Training Avanzato (+1.03%)**: L'introduzione di tecniche moderne di Data Augmentation (AutoAugment, Cutout) ha sbloccato un ulteriore margine, portando il risultato finale a **93.50%**. Questo dimostra che l'architettura ha ancora "capacità di apprendimento" (headroom) inutilizzata e beneficia enormemente dalla regolarizzazione forte.

## Verdetto Finale per la Tesi
La combinazione proposta (**MobileNetECA Reparameterized**) si dimostra non solo efficiente, ma estremamente robusta.
Con un addestramento avanzato, raggiunge il **93.50%**, un risultato eccezionale per un modello con soli 76.6k parametri, superando nettarmente la baseline e ponendosi come riferimento per l'efficienza su CIFAR-10.
Le tre chiavi del successo sono state:
1.  **Design**: ECA + GELU.
2.  **Ottimizzazione**: Structural Reparameterization.
3.  **Regolarizzazione**: Strong Data Augmentation.
