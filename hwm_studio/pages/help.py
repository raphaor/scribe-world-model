"""Page d'aide — HWM Studio.

Documentation interactive de l'interface, organisée en sections
repliables. Couvre le flux de travail complet : sources de données,
entraînement, évaluation, visualisation, annotation et dépannage.
"""

from __future__ import annotations

from nicegui import ui


def build():
    """Construit la page d'aide."""
    ui.label("Aide").classes("text-h4 q-mb-md")

    _section_bienvenue()
    _section_sources()
    _section_entrainement()
    _section_evaluation()
    _section_visualisation()
    _section_annotation()
    _section_modeles()
    _section_lm()
    _section_depannage()


# --------------------------------------------------------------------------- #
#  Sections
# --------------------------------------------------------------------------- #

def _section_bienvenue():
    with ui.expansion("Bienvenue", icon="waving_hand").classes("w-full").props("default-opened"):
        ui.markdown("""
**HWM Studio** est une interface web locale pour entraîner, évaluer et annoter
des modèles de reconnaissance d'écriture manuscrite (OCR) pour documents
historiques. Elle enveloppe les scripts Python du projet
(`train.py`, `recognize.py`, `visualize.py`, `annotate.py`, `train_lm.py`)
dans une interface graphique accessible.

### Lancer l'application

```bash
cd scribe-world-model
python -m hwm_studio.app
```

L'interface s'ouvre dans le navigateur à l'adresse
**http://localhost:8080**.

### Flux de travail typique

1. **Sources de données** — sélectionner les répertoires ALTO à utiliser
2. **Entraînement** — lancer un entraînement (ou reprendre d'un checkpoint)
3. **Évaluation CER** — mesurer le taux d'erreur
4. **Visualisation** — inspecter visuellement les prédictions
5. **Annotation** — corriger la ground truth pour le prochain cycle
""").classes("w-full")


def _section_sources():
    with ui.expansion("Sources de données", icon="folder_open").classes("w-full"):
        ui.markdown("""
### Répertoires ALTO

Un répertoire ALTO contient des fichiers `.xml` (segmentation des lignes de
texte) et `.jpg` (scans des pages), appariés par nom de fichier. Le fichier
`METS.xml` est automatiquement ignoré.

### Gérer le catalogue

- **Auto-scan** : au démarrage, l'application scanne `/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/`
  et propose les sous-dossiers contenant des `.xml`.
- **Ajout manuel** : bouton **+ Ajouter** pour saisir un chemin personnalisé.
- **Suppression** : icône corbeille sur chaque carte.

Chaque source peut être activée/désactivée par une case à cocher. Seules les
sources cochées sont transmises aux scripts.

### Mode Original / GT augmentée

C'est la fonctionnalité clé de la gestion des données :
""").classes("w-full")

        with ui.card().classes("bg-orange-1-10 w-full"):
            ui.markdown("""
**● Original** — les scripts lisent les fichiers `.xml` d'origine tels que
fournis par Kraken. C'est le mode par défaut.

**● GT augmentée** — l'application crée un répertoire de **staging** où les
fichiers `_gt.xml` (corrigés via l'annotation) remplacent les `.xml` d'origine.
Les `.jpg` sont liés par symlink (pas de duplication). Le répertoire de staging
est passé à la place du chemin original.

> **Les fichiers d'origine ne sont jamais modifiés.** Le staging est recréé
> à la volée si les `_gt.xml` changent.
""").classes("w-full")

        ui.markdown("""
### Couverture de ground truth

La barre de progression indique le ratio de fichiers `_gt.xml` par rapport aux
fichiers `.xml` dans chaque répertoire. Une couverture de 100% signifie que
toutes les pages ont été annotées.
""").classes("w-full")


def _section_entrainement():
    with ui.expansion("Entraînement", icon="model_training").classes("w-full"):
        ui.markdown("""
### Trois modes d'entraînement

| Mode | Description | Pertes actives |
|------|-------------|----------------|
| **mixed** (défaut) | Alterne batches supervisés et auto-supervisés | CTC + JEPA + SIGReg |
| **full** | Pur supervisé, toutes les pertes | CTC + JEPA + SIGReg |
| **adapt** | Pur auto-supervisé, sans labels | JEPA + SIGReg uniquement |

### Paramètres essentiels

- **Version** : architecture du modèle (v2 à v18). Chaque version a ses
  hyperparamètres figés dans `config.py` pour la reproductibilité.
- **Epochs** : nombre de passages sur les données (défaut : 30).
- **Batch size** : lignes par batch (défaut : 32). Réduire en cas d'OOM.
- **LR** : taux d'apprentissage (défaut : 1e-3).
- **Checkpoint** : reprendre depuis un `.pt` existant (optionnel).
- **Save path** : chemin de sauvegarde du checkpoint.

### Paramètres avancés

- **No JEPA** : désactive la branche auto-supervisée (baseline CTC-only).
- **No AMP** : désactive la mixed-precision float16 (recommandé pour v15+).
- **Encoder LR mult** : le tronc (encoder) apprend plus lentement que la tête
  CTC. Défaut : 0.1× . Mettre 1.0 pour désactiver le LR discriminatif.
- **Warmup epochs** : montée en rampe linéaire du LR au démarrage
  (auto-activé à 2 lors d'un changement de phase).
- **Freeze encoder** : gèle l'encoder pendant N epochs pour stabiliser la
  tête CTC (style ULMFiT).
- **Lambda pred / Lambda sigreg** : surcharge les poids de perte JEPA et
  SIGReg. Laisser vide pour utiliser les valeurs par défaut du modèle.
- **Constant LR** : pas de décroissance cosinus (recette ketos/Lectaurep).
""").classes("w-full")

        with ui.card().classes("bg-blue-1-10 w-full"):
            ui.markdown("""
### Curriculum learning (adapt → full)

Workflow classique en deux phases :

1. **Phase 1** : `--mode adapt --save-path hwm_v17_adapt.pt`
   — pré-entraîne l'encoder sans labels.
2. **Phase 2** : `--checkpoint hwm_v17_adapt.pt --mode full --save-path hwm_v17_full.pt`
   — ajoute la tête CTC et fine-tune.

Le détecteur de **phase switch** réinitialise automatiquement l'optimiseur,
le scheduler et active un warmup de 2 epochs quand le mode change.
""").classes("w-full")

        with ui.card().classes("bg-blue-1-10 w-full"):
            ui.markdown("""
### Protocole held-out scribe

Pour mesurer la généralisation à un scripteur inconnu :

1. Sélectionner tous les répertoires sauf un dans **Sources**.
2. Entraîner en mode `adapt` (pré-apprentissage SSL).
3. Désélectionner tout sauf le répertoire tenu de côté.
4. Fine-tuner en mode `full` depuis le checkpoint.

Le save-path peut être personnalisé, par exemple `hwm_v18_bars.pt`.
""").classes("w-full")


def _section_evaluation():
    with ui.expansion("Évaluation CER", icon="fact_check").classes("w-full"):
        ui.markdown("""
### CER (Character Error Rate)

Le CER est la distance d'édition (Levenshtein) normalisée par la longueur du
texte de référence. Plus il est bas, meilleur est le modèle.

### Splits

| Split | Description |
|-------|-------------|
| **val** (défaut) | 20% des données, seed 42 — partition officielle |
| **train** | 80% des données (pour vérifier l'overfitting) |
| **all** | Toutes les données (sans split) |

### Greedy vs Beam search

- **Greedy** : décodage argmax rapide, 1 seule passe.
- **Beam search** : explore plusieurs hypothèses en parallèle. Plus lent
  mais généralement meilleur. Largeur du beam défaut : 20.

### Modèle de langue (LM)

Un n-gramme de caractères (fichiers `.pkl`) peut biaiser le beam search
pour privilégier des séquences vraisemblables. Poids défaut : 0.3.

- **LM weight = 0** : pure beam search CTC.
- **LM weight = 1** : le LM domine.

### Mode compare

Lance le greedy ET le beam search, puis affiche les deux CER avec le delta.
Idéal pour mesurer l'apport du beam search + LM.
""").classes("w-full")


def _section_visualisation():
    with ui.expansion("Visualisation", icon="image").classes("w-full"):
        ui.markdown("""
### Deux modes

- **Validation** : utilise les sources sélectionnées avec le split seedé
  (identique à `recognize.py`).
- **Fichier / Dossier** : spécifier un chemin vers un `.xml` ou un
  répertoire. Toutes les lignes sont affichées (pas de split).

### Lecture des couleurs de confiance

Chaque caractère prédit est coloré selon sa probabilité :

- 🟢 **Vert** : haute confiance
- 🟡 **Jaune** : confiance moyenne
- 🔴 **Rouge** : faible confiance

Les caractères erronés (vs la ground truth) sont soulignés en rouge.

### Options

- **Sort by CER** : trie les pires prédictions en premier.
- **Top N** : limite aux N pires cas.
- **No GT** : prédit toutes les lignes même sans transcription.

> ⚠️ Cette commande ouvre une **fenêtre matplotlib externe**. Naviguez avec
> les flèches ← → ou les boutons Prev/Next.
""").classes("w-full")


def _section_annotation():
    with ui.expansion("Annotation", icon="edit_note").classes("w-full"):
        ui.markdown("""
### Deux modes

| Mode | Lignes affichées | Pré-remplissage |
|------|------------------|-----------------|
| **annotate** | Sans ground truth (vides) | Prédiction du modèle |
| **review** | Avec ground truth existante | GT actuelle pour correction |

### Fichiers `_gt.xml`

L'annotation crée des fichiers `*_gt.xml` **à côté** des fichiers `.xml`
d'origine. Les originaux ne sont **jamais modifiés**.

Un fichier `*_gt.progress.json` mémorise les lignes acceptées et la dernière
page consultée, ce qui permet de **reprendre une session** d'annotation.

### Comment la GT remonte à l'entraînement

Par défaut, les fichiers `_gt.xml` sont **invisibles** au training (le
pipeline apparie `.xml` et `.jpg` par nom de base ; `page_gt.xml` cherche
`page_gt.jpg` qui n'existe pas).

Pour utiliser la GT corrigée lors d'un entraînement :

1. Allez dans **Sources de données**.
2. Activez le toggle **GT augmentée** sur le répertoire concerné.
3. Lancez l'entraînement — l'application crée automatiquement un
   répertoire de staging où les `_gt.xml` remplacent les `.xml`.

### Options beam search

L'annotation peut utiliser le beam search (+ LM) pour de meilleures
prédictions initiales. Activer l'option dans le formulaire.
""").classes("w-full")


def _section_modeles():
    with ui.expansion("Modèles (checkpoints)", icon="storage").classes("w-full"):
        ui.markdown("""
### Browser de checkpoints

La page Modèles liste tous les fichiers `hwm_*.pt` à la racine du projet
avec leurs métadonnées : version, mode, epoch, loss, taille, alphabet.

### Tags automatiques

Les suffixes des noms de fichiers sont parsés et affichés comme tags :

- `hwm_v11_adapt.pt` → tag **adapt**
- `hwm_v18_322_bars.pt` → tags **bars**
- `hwm_v14_nojepa.pt` → tag **nojepa**

### Actions

- **Évaluer** : redirige vers la page Évaluation CER.
- **Visualiser** : redirige vers la page Visualisation.
- **Fine-tuner** : redirige vers la page Entraînement (sélectionner le
  checkpoint dans le champ correspondant).
- **Supprimer** : supprime le fichier `.pt` après confirmation.

### Convention de nommage

| Suffixe | Signification |
|---------|---------------|
| `_adapt` | Phase auto-supervisée |
| `_full` | Phase supervisée |
| `_nojepa` | Branche JEPA désactivée |
| `_bars`, `_stcham` | Corpus spécifique (Dordogne) |
| `_hard` | Perturbations renforcées |
""").classes("w-full")


def _section_lm():
    with ui.expansion("Langue (LM n-gramme)", icon="translate").classes("w-full"):
        ui.markdown("""
### À quoi sert un modèle de langue ?

Un n-gramme de caractères capture les régularités statistiques des
transcriptions (ex: « qu » est plus fréquent que « qz »). Il sert à
**biaiser le beam search** vers des séquences plus vraisemblables.

### Entraîner un LM

1. Sélectionner les sources de données (les textes d'entraînement sont
   extraits du split seedé, identique à `train.py`).
2. Choisir la version du modèle (pour `img_height` et `cnn_width_stride`).
3. Indiquer le fichier de sortie (ex: `char_8gram.pkl`).
4. Choisir l'ordre n (défaut : 8 — un bon compromis).

### Utiliser le LM

Le fichier `.pkl` peut être chargé dans les pages **Évaluation**,
**Visualisation** et **Annotation** en activant le beam search puis en
sélectionnant le LM dans le champ dédié.
""").classes("w-full")


def _section_depannage():
    with ui.expansion("Dépannage", icon="build").classes("w-full"):
        ui.markdown("""
### Le bouton « Lancer » est désactivé / rien ne se passe

Vérifiez qu'**aucun autre processus n'est en cours**. La barre d'état en bas
de page indique le statut. Un seul processus à la fois est autorisé.
Utilisez le bouton **Stop** pour interrompre.

### « CUDA out of memory »

La VRAM est épuisée. Solutions :

- Réduire le **batch size** (ex: passer de 32 à 16 ou 8).
- Activer **No bucket** (désactive le regroupement par largeur).
- Activer **Grad checkpoint** (v12+, échange du compute contre de la VRAM).
- Vérifier qu'aucun autre processus n'utilise le GPU (`nvidia-smi`).

### « Un processus est déjà en cours »

Un subprocess tourne encore. Soit attendre sa fin, soit cliquer sur
**Stop** dans la barre d'état. Si le processus est zombie (planté sans
libérer le verrou), redémarrer l'application.

### Le GPU n'apparaît pas dans le Dashboard

- Vérifier que `nvidia-smi` est accessible (`where nvidia-smi` dans un
  terminal).
- Le GPU s'affiche uniquement si PyTorch détecte CUDA.

### Les répertoires de données ne s'affichent pas

- Vérifier que les chemins existent et sont accessibles.
- Utiliser **+ Ajouter** pour ajouter manuellement un chemin.
- Le scan auto ne couvre que `/media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/` par défaut.

### Le mode GT augmentée ne marche pas

- Vérifier qu'il existe des fichiers `_gt.xml` dans le répertoire
  (le toggle GT n'apparaît que si `gt_count > 0`).
- Sur Windows, les symlinks nécessitent le **Mode développeur** activé.
  Sans cela, l'application copie les JPG (plus lent mais fonctionnel).

### Ne pas supprimer `.cache_alto/` pendant un entraînement

Les fichiers `.cache_alto/*.pkl` sont lus en direct pendant le training.
Les supprimer en plein epoch provoque un crash.
""").classes("w-full")
