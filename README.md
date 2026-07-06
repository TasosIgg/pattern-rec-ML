# Αναγνώριση Προτύπων & Μηχανική Μάθηση — Εργασία 2 (CNN)

**Ονοματεπώνυμο:** Αναστάσιος Ιγγλεζάκης
**ΑΜ:** 1115202300054

## Περιγραφή

Το notebook `CNN.ipynb` υλοποιεί ταξινόμηση έργων τέχνης (WikiArt, dataset `wikiart_hw2.npz`) σε 4 καλλιτεχνικά κινήματα — Μπαρόκ, Ιμπρεσιονισμός, Κυβισμός, Αφαιρετικός Εξπρεσιονισμός — από εικόνες 32×32×3. Συγκρίνονται τρεις προσεγγίσεις: απλό FFN σε flat pixels, CNN εκπαιδευμένο από το μηδέν, και transfer learning με frozen ResNet18 (linear probe).

## Δεδομένα

- 5.400 εικόνες (uint8, 32×32×3), ισορροπημένες: 1.350 ανά κλάση.
- Κανονικοποίηση στο [0,1] και μετατροπή σε channels-first (N,C,H,W).
- Stratified split: **Train 3.200 / Validation 800 / Test 1.400**.
- DataLoaders με batch size 16.

Το αρχείο `wikiart_hw2.npz.zip` αναμένεται στο `/content/` (Colab) και αποσυμπιέζεται από το notebook.

## Δομή του notebook

### §0 — Προετοιμασία & εξερεύνηση
- Seeds για αναπαραγωγιμότητα (`SEED=42`, deterministic cuDNN).
- Φόρτωση/κανονικοποίηση δεδομένων, splits, οπτικοποίηση 2 τυχαίων εικόνων ανά κίνημα (**Q0.1**: ποιοτική ανάλυση).

### §1 — FFN baseline
- Αρχιτεκτονική: `Flatten → 3072→128→32→4` με ReLU (397.604 παράμετροι).
- Εκπαίδευση 30 εποχές με Adam (lr=0.001), CrossEntropyLoss· επιλογή best model βάσει validation F1 macro.
- Σύγκριση χρόνου CPU vs GPU (speedup ~3.4x).
- Αξιολόγηση στο test set (Acc ~51.8%, F1 ~51.5%), confusion matrix, οπτικοποίηση βαρών πρώτου επιπέδου ως RGB εικόνες (**Q1.1**, **Q1.2**).

### §2 — CNN από το μηδέν
- Παραμετρική κλάση `CNN`: 4 conv layers (κανάλια 3→16→32→64→128, kernel 5×5) + FC head (→1024→256→32→4), με επιλογές padding / max-pooling / ReLU activations.
- Ablation τριών εκδόσεων:

  | Έκδοση | Ρύθμιση | Παράμετροι | Test F1 |
  |--------|---------|-----------:|--------:|
  | §2.1 | χωρίς padding/pooling/activations | 34M | ~36.5% |
  | §2.2 | + padding=2 και MaxPool2d(2) | 1.07M | ~57.9% |
  | §2.3 | + ReLU activations | 1.07M | ~56.0% |

- Σύγκριση με FFN, confusion matrices (**Q2.1**, **Q2.2**, **Q2.3**).

### §3 — Βελτιστοποίηση εκπαίδευσης
- **§3.1** Σύγκριση optimizers (SGD, SGD+momentum, Adam, RMSprop, AdamW) με ίδιο seed· καλύτερος ο AdamW (F1 ~58.8%). Επανεκπαίδευση με CosineAnnealingLR scheduler.
- **§3.2** `CNN_BN` (BatchNorm μετά από κάθε conv) και `CNN_BN_Drop` (+Dropout 0.3 στο FC head). Πείραμα 2×2, 60 εποχές:
  - baseline
  - μόνο data augmentation (RandomHorizontalFlip, RandomCrop με padding=4, ColorJitter)
  - μόνο regularization (dropout + weight decay 1e-4)
  - **augmentation + regularization** ← καλύτερο, F1 ~72.3%

  Ανάλυση overfitting μέσω train–val loss gap (**Q3.1**, **Q3.2**).

### §4 — Transfer learning: ResNet18 linear probe
- Προεκπαιδευμένο ResNet18 (`IMAGENET1K_V1`) με παγωμένα βάρη· εκπαιδεύεται μόνο νέο FC layer 512→4 (2.052 παράμετροι).
- Upscale εικόνων 32→224 (bilinear) και ImageNet normalization.
- 10 εποχές με Adam· Test Acc ~75.5%, F1 ~75.3% — καλύτερο όλων παρά τις ελάχιστες εκπαιδεύσιμες παραμέτρους (**Q4.1**).

### §5 — Ερμηνευσιμότητα
- Οπτικοποίηση φίλτρων πρώτου conv layer: CNN §3.2 (16×5×5) vs ResNet18 (16 πρώτα από 64×7×7).
- Activation maps πρώτου layer για ένα δείγμα ανά κλάση, recap των FFN βαρών για σύγκριση (**Q5.1**).

## Συνοπτικά αποτελέσματα (test set)

| Μοντέλο | Test Acc | F1 macro | Trainable params |
|---------|---------:|---------:|-----------------:|
| FFN flat pixels (§1) | 51.79% | 51.47% | 397.604 |
| CNN βέλτιστο (§3.2) | — | 72.32% | 1.066.308 |
| Frozen ResNet18 (§4) | 75.50% | 75.25% | 2.052 |

## Εξαρτήσεις

```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

## Εκτέλεση

Τρέξτε τα cells του `CNN.ipynb` σειριακά (τα πειράματα εξαρτώνται από προηγούμενα cells). Εκτιμώμενος χρόνος εκτέλεσης σε GPU (T4): **~65 λεπτά**.
