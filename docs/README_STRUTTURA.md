# README - Struttura Tesi

Questa directory contiene tutti i materiali di supporto per scrivere la tesi triennale sulla classificazione CIFAR-10 con il modello Mimir.

## 📂 Struttura Directory

```
/Users/dima/Desktop/Tesi/
├── docs/
│   ├── template_capitoli/      # Template dettagliati per ogni capitolo
│   │   ├── capitolo_1_introduzione.md
│   │   ├── capitolo_3_metodologia.md
│   │   └── capitolo_4_risultati.md
│   ├── bibliografia/           # File BibTeX con riferimenti
│   │   └── references.bib
│   ├── grafici/               # Grafici generati per la tesi
│   ├── checklist_tesi.md      # Checklist completa step-by-step
│   └── GUIDA_TESI_UNICA.md    # Guida rapida iniziale
├── src/
│   ├── legacy/
│   │   └── pareto_mimir7.py   # Implementazione modello Mimir
│   └── scripts/
│       ├── calcola_metriche_tesi.py    # Analisi metriche
│       └── genera_grafici_tesi.py      # Script generazione grafici
├── models/                     # Checkpoint modelli salvati (.pt)
└── brain/                      # Artifacts di pianificazione
    ├── task.md
    ├── implementation_plan.md
    └── guida_pratica.md
```

## 🚀 Come Iniziare

### 1. Prima di Scrivere

**Raccogliere i dati finali** (quando avrai GPU):
```bash
cd /Users/dima/Desktop/Tesi
# Re-eseguire training se necessario o verificare risultati
python src/legacy/pareto_mimir7.py
```

**Verificare metriche**:
```bash
python src/scripts/calcola_metriche_tesi.py
```

**Generare grafici**:
```bash
python src/scripts/genera_grafici_tesi.py
```

### 2. Durante la Scrittura

**Segui i template** in `docs/template_capitoli/`:
- Ogni template ha paragrafi esempio
- Bullet points su cosa scrivere in ogni sezione
- Tabelle e figure da includere

**Ordine consigliato**:
1. Cap. 3 (Metodologia) - hai il codice, è facile
2. Cap. 4 (Risultati) - hai i dati
3. Cap. 2 (Stato dell'Arte) - leggere paper
4. Cap. 5 (Conclusioni) - sintetizzare
5. Cap. 1 (Introduzione) - overview generale

### 3. Gestione Citazioni

Il file `docs/bibliografia/references.bib` contiene già i paper principali.

**In LaTeX**:
```latex
\cite{mobilenetv2}  % Per citare MobileNetV2
\cite{ecanet}       % Per citare ECA-Net
```

**Aggiungere nuove citazioni**: copia il formato da Google Scholar > Cite > BibTeX

## 📊 Grafici

Lo script `src/scripts/genera_grafici_tesi.py` genera:

1. **training_curve.png**: Loss e accuracy per 50 epoche
2. **accuracy_vs_params.png**: Scatter plot confronto modelli
3. **model_comparison_bar.png**: Bar chart dimensioni
4. **confusion_matrix.png**: Confusion matrix Mimir

⚠️ **NOTA**: Alcuni grafici usano dati esempio - sostituisci con dati reali!

## 📝 Checklist

Usa `docs/checklist_tesi.md` per tracciare il progresso:
- [ ] Tick quando completi ogni sezione
- [ ] Revisione ortografica
- [ ] Controllo citazioni
- [ ] Generazione PDF finale

## 📚 Risorse Utili

**Paper Chiave** (già in references.bib):
- MobileNetV2 (Sandler et al., 2018)
- ECA-Net (Wang et al., 2020)
- SE-Net (Hu et al., 2018)
- ResNet (He et al., 2016)

**Dataset**:
- CIFAR-10 (Krizhevsky, 2009)

## 🎯 Obiettivi Tesi

- **Modello**: Mimir (~200k params, 88% acc)
- **Confronto**: MobileNetV2, ShuffleNetV2, ResNet-20
- **Focus**: Edge AI, efficienza, trade-off accuracy/size
- **Lunghezza**: 40-50 pagine

## ⏱️ Timeline (4 settimane)

- **Week 1**: Dati + Cap 3, 4
- **Week 2**: Cap 2, 5
- **Week 3**: Cap 1, Abstract
- **Week 4**: Revisione + PDF finale

## 💡 Tips

1. **Scrivi metodologia per prima** - è la più facile
2. **Ogni figura DEVE essere citata** nel testo
3. **Acronimi**: spiega alla prima occorrenza (ECA = Efficient Channel Attention)
4. **Usa tono professionale**: "si è sviluppato", non "ho fatto"
5. **LaTeX**: usa Overleaf per editing collaborativo

## 🔗 Links Utili

- Template LaTeX tesi: (chiedi al dipartimento)
- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- PyTorch docs: https://pytorch.org/docs/

## 📞 Supporto

Per domande sul codice o struttura, rivedi:
- `brain/guida_pratica.md` - guida super-concisa
- `brain/implementation_plan.md` - piano dettagliato completo
- Template capitoli - esempi paragrafi

---

**Ultimo aggiornamento**: 2026-02-12  
**Versione**: 1.0
