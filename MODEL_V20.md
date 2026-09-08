# HWM v20 — LeVJEPA sur CNN renforcé : comparateur CLS vs ALL

Baseline : **v18 Val CER 10.3%** (CTC-pure). v19 = squelette SSL LeVJEPA (pure),
CNNs relégués. v20 récupère les CNNs — indispensables — et empile **dessus** la
construction v19 : transformer block-causal + token `[cls]` + réseau dense `h_phi`
qui compare sans collapse (SIGReg Epps-Pulley).

Fichiers : `model.py` (classe `HWMv20`), `config.py` (bloc `*_V20`), `model_registry.py`
(`_build_v20` + entrée `"v20"`), `train.py` (`--compare-mode`). Enregistré, `hwm_v20.pt`.

## Architecture (2 temps)

```
ligne (B, 64, W)
  -> stem CNN KRAKEN renforcé (STEM_CHANNELS_V20=128, kernels larges
     horizontaux, 2xMaxPool /4)            // le « on reprend les cnn »
  -> patches carrés 16x16 -> 1 token = 64 px
  -> fenêtres coulissantes K=8, stride 4 (chevauchement 50%)
  -> transformer block-causal 8x d=256 ff=1024 RoPE, [cls] readout
     (attends à tous, personne ne lui attend), LayerNorm finale

CTC  (phase full) : features par position (sortie LayerNorm, PAS le [cls])
  -> LSTM LÉGER 1xBiLSTM(320) -> CTC.   // espoir : le transformer fait le
     boulot, on ne veut pas un LSTM lourd (bump à 2 si le CER le justifie)

SSL  (phase adapt) : projecteur h_phi Linear(256->2048)->LayerNorm->GELU
  ->Linear(2048->128) sur la représentation choisie. JETÉ après pré-entr.
```

## Objectif SSL (fidélité LeVJEPA, pas de stop-grad/EMA/predictor)

- 1 vue **globale** + V=4 vues **locales** (recadrage horizontal + photométrie).
- `L_inv = 1/(V+1) Σ_v ‖z_global − z_locale‖²` — **gradients à travers les deux
  branches** (le CNN et le transformer apprennent des deux côtés).
- **SIGReg Epps-Pulley sur le batch des [cls]** (toutes vues), seul hp SSL
  (`--lambda-sigreg`, défaut 0.1). Anti-collapse provable, co-localisé comme au papier.
- **Masquage indépendant par vue** : chaque vue jette ses propres tokens
  (jamais le [cls], jamais le patch 0 — ancre). Taux balayé via
  `--token-drop` (défaut 0.5 ; 95% = régime vidéo, probablement trop pour
  des lignes HTR — à tester).

## Le test central — `--compare-mode`

Que compare-t-on ? Deux représentations, la même machine :
- `cls` (défaut) : compare les embeddings `[cls]` (comportement v19), un readout
  global de la ligne entière ;
- `all`  : compare les features **tout-tokens moyennées** (sortie LayerNorm) —
  la représentation *par position* que la CTC consomme ensuite.

Dans les deux cas SIGReg reste sur le batch des `[cls]` (fidélité paper). La
question : le readout global suffit-il, ou faut-il la représentation par position
dont le CTC a besoin ? (Le skill v19 — « CTC needs per-position features, pas
juste un [cls] » — plaide a priori pour `all`, mais on mesure.)

## Commandes

```bash
python train.py --model-version v20 --mode adapt --save-path hwm_v20_adapt.pt
python train.py --model-version v20 --mode full  --checkpoint hwm_v20_adapt.pt
# variantes / ablations
python train.py --model-version v20 --mode adapt --save-path hwm_v20_adapt_all.pt --compare-mode all
python train.py --model-version v20 --mode adapt --save-path hwm_v20_adapt_d30.pt --token-drop 0.3
```

## Matrice de test (ordre d'information)

1. `--compare-mode` : `cls` (baseline v19) vs `all` — est-ce que `all` sauve la
   représentation par position ?
2. `--token-drop` 0/0.3/0.5/0.7 : le 95% vidéo est-il trop pour des lignes HTR ?
3. Profondeur LSTM 1 vs 2 (phase full) : le transformer allège-t-il le LSTM ?
4. `--lambda-sigreg` 0 vs 0.1 : SIGReg est-il actif ou inerte ici (le canari v17/v19) ?

## Smoke test (Pi, CPU, validé)

`_build_v20` instancie ; forward `(2,64,384)->z_seq (2,6,640), ctc (2,6,30)` ;
adapt loss 0.434 avec gradient jusqu'à l'image ; full loss 3.88 ; `compare_mode=all`
adapt 0.383. **17.1M params** (stem 128). Exit code 0.

## Décisions & risques (conversation du jour)

- **Pas de stop-grad** : voulu par le user (« que le transformer et le cnn
  apprennent par les deux branches »), LeVJEPA le fait nativement via `L_inv`.
  Le **seul** garde-fou anti-collapse est donc SIGReg — posé comme au papier
  (sur le batch [cls]). Canari : signature v19 (inv→0, sigreg épinglé au plateau)
  = collapse ; sinon sain.
- **SIGReg sur le [cls], pas les deux z** : c'est une correction à ma suggestion
  initiale (SIGReg sur les deux z). On suit le papier, un seul λ.
- **Stem renforcé à 128** : récupère la capacité CNN sans casser la tokenisation
  (img_height//4 == patch exigé par l'encodeur).
- Risque ouvert : `compare_mode='all'` compare des features moyennées — le moyennage
  peut diluer l'information positionnelle au moment du comparé ; mais c'est justement
  ce qu'on veut mesurer, et CTC reste sur les features par position de toute façon.