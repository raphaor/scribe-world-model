# Comparaison architecturale : HWM v4 vs Kraken (defaut)

## Vue d'ensemble

| | Kraken (defaut) | HWM v4 |
|---|---|---|
| **Approche** | Ligne entiere en 1D (colonnes) | Frames decoupees + world model |
| **Entree** | 1x120xW (ligne complete) | 1x48x32 (frames, stride 4) |
| **Parametres** | ~4.0M | ~8.3M |
| **Objectif** | OCR pur (CTC) | Self-supervised (pred+SIGReg) + OCR (CTC) |

## Partie convolutive

### Kraken
```
Conv2D 3x13, 32 filtres + ReLU + Dropout(0.1) + MaxPool(2,2)
Conv2D 3x13, 32 filtres + ReLU + Dropout(0.1) + MaxPool(2,2)
Conv2D 3x9,  64 filtres + ReLU + Dropout(0.1) + MaxPool(2,2)
Conv2D 3x9,  64 filtres + ReLU + Dropout(0.1)
```
- 4 couches, ~207K params
- Kernels **rectangulaires** (3x13, 3x9) : capturent des traits horizontaux larges
- Pas de BatchNorm, regularisation par Dropout seul
- Entree haute resolution (120px) sur la ligne entiere

### HWM v4
```
Conv2D 3x3,  64 filtres  + BN + ReLU + MaxPool(2,2)
Conv2D 3x3, 128 filtres  + BN + ReLU + MaxPool(2,2)
Conv2D 3x3, 256 filtres  + BN + ReLU
Conv2D 3x3, 512 filtres  + BN + ReLU + MaxPool(2,2)
Conv2D 3x3, 512 filtres  + BN + ReLU + AdaptiveAvgPool(4,2)
Linear(4096, 256) + LayerNorm
```
- 5 couches + projection, ~5.0M params
- Kernels **carres** (3x3) : champ receptif horizontal plus etroit
- BatchNorm + LayerNorm en sortie
- Entree par frame (48x32px), pas la ligne entiere

## Partie sequentielle / contextuelle

### Kraken
```
BiLSTM 200 hidden (960 -> 400) + Dropout(0.1)
BiLSTM 200 hidden (400 -> 400) + Dropout(0.1)
BiLSTM 200 hidden (400 -> 400) + Dropout(0.5)
```
- **3 couches BiLSTM empilees** = ~3.8M params
- Chaque couche raffine la comprehension contextuelle
- Voit la ligne entiere dans les deux directions
- C'est le coeur de la puissance de Kraken

### HWM v4
```
Transformer causal 4 couches, 8 tetes, ff=512 (pour prediction)
BiLSTM 256 hidden, 1 couche (pour CTC uniquement)
```
- Predictor : ~2.2M params (utilise pour l'objectif self-supervised, pas le CTC)
- CTC BiLSTM : ~1.1M params, **1 seule couche** vs 3 chez Kraken

## Tete CTC

| | Kraken | HWM v4 |
|---|---|---|
| **Entree** | 400 dims (sortie 3 BiLSTM) | 512 dims (sortie 1 BiLSTM) |
| **Projection** | Linear(400, C) | Linear(512, C) |
| **Params** | ~40K | ~1.1M (BiLSTM inclus) |

## Repartition des parametres

### Kraken (~4.0M total)
```
Conv layers:    207K   ( 5%)
3x BiLSTM:    3,786K   (94%)
CTC head:        40K   ( 1%)
```

### HWM v4 (~8.3M total)
```
Encoder CNN:  4,962K   (60%)
Predictor:    2,174K   (26%)
CTC BiLSTM:   1,119K   (14%)
```

## Observations

1. **Kraken investit 94% de ses parametres dans les BiLSTM.** Sa partie convolutive
   est legere (207K) — les convolutions extraient des features simples, et les 3 couches
   BiLSTM font le gros du travail de contextualisation et reconnaissance.

2. **HWM v4 investit 60% dans l'encodeur CNN.** C'est coherent avec l'objectif
   self-supervised (predire la frame suivante) qui demande des embeddings riches.
   Mais le CTC n'a qu'un seul BiLSTM pour contextualiser.

3. **Les kernels rectangulaires de Kraken** (3x13) sont concus pour capturer la
   structure horizontale de l'ecriture (liaisons, traits). Nos kernels 3x3 ont un
   champ receptif horizontal plus petit, compense partiellement par la profondeur.

4. **L'approche est fondamentalement differente** : Kraken traite la ligne comme
   un flux 1D continu. HWM decoupe en frames et reconstruit le contexte via le
   transformer (pour pred) et le BiLSTM (pour CTC). Le transformer causal ne
   contribue pas directement a la reconnaissance — il sert l'objectif self-supervised.

## Pistes d'amelioration potentielles

- Augmenter le nombre de couches BiLSTM dans le CTC head (1 -> 2 ou 3)
- Tester des kernels rectangulaires dans l'encodeur (3x7 ou 3x9)
- Augmenter la hauteur d'entree (48 -> 64 ou 80px)
- Alimenter le CTC head avec les sorties du transformer en plus des embeddings