# HWM-v2 : Plan d'implementation detaille

## Vision

Creer une architecture hybride LeWorldModel + Kraken pour la reconnaissance
d'ecriture manuscrite sur registres d'etat civil anciens.

Le probleme central : Kraken necessite un fine-tuning supervise pour chaque
nouveau scripteur. Notre approche separe l'adaptation (auto-supervisee, sans
labels) de la reconnaissance (decodeur CTC standard), permettant une
acclimatation a un nouveau scripteur sans transcription.

```
Image de ligne --> [CNN 2D Encoder] --> Embeddings (B, T, D)
                                              |
                             +----------------+----------------+
                             |                                 |
                  [Transformer Predictor]          [CTC Recognition Head]
                  (next embedding, causal)         (character logits)
                             |                                 |
                    MSE + SIGReg loss                      CTC loss
                   (auto-supervise)                      (supervise)
```

Loss totale : `L = L_pred + lambda_sigreg * L_sigreg + lambda_ctc * L_ctc`

Workflow sur un nouveau registre :
1. Auto-adaptation : quelques epochs de prediction d'embeddings sur les pages
   brutes (aucune transcription)
2. Decodage : la tete CTC (figee ou legerement ajustee) transcrit les lignes

---

## Prerequis : Environnement conda dedie

Creer un environnement conda dedie au projet.

```bash
conda create -n scribe python=3.11 -y
conda activate scribe
pip install torch torchvision numpy pillow
pip install kraken
```

Verifier :
```bash
python -c "import torch; print(torch.__version__)"
python -c "from kraken.lib.xml import XMLPage; print('OK')"
python -c "from kraken.lib.segmentation import extract_polygons; print('OK')"
python -c "from kraken.containers import BaselineLine, Segmentation; print('OK')"
python -c "from kraken.lib.codec import PytorchCodec; print('OK')"
```

Si kraken pose probleme a l'install (compilation de dependances sur Windows),
l'alternative est `pip install kraken --no-deps` puis installer manuellement
les sous-dependances necessaires (scipy, scikit-learn, shapely, lxml).

Toutes les commandes du plan utilisent : `conda run -n scribe python ...`

---

## Donnees source

Les donnees annotees sont dans `D:\OCR_genealogie\Alto\` :

```
D:\OCR_genealogie\Alto\
  bars_dordogne_alto\              (45 pages, ~1000+ lignes)
  saint_chamassy_dordogne_alto_set_1\    (77 pages)
  saint_chamassy_dordogne_alto_set_train\ (34 pages)
```

Chaque page = 1 paire :
- `*.jpg` : scan pleine page (~2500x2300 px)
- `*.xml` : ALTO XML avec pour chaque ligne :
  - `BASELINE` : coordonnees polyline (pixels)
  - `Shape > Polygon` : contour polygonal
  - `String CONTENT` : transcription (ex: "Le 6 Janvier 1744 A baptize...")

Alphabet reel : majuscules, minuscules, chiffres, ponctuation, accents,
caracteres speciaux (¥ comme marqueur, &#x27; pour apostrophe).
L'alphabet sera construit dynamiquement a partir des donnees (via
PytorchCodec de kraken ou manuellement).

---

## Phase 0 : Chargeur de donnees ALTO (nouveau fichier)

**Objectif** : charger les donnees reelles en reutilisant les parsers kraken.

### Nouveau fichier : `data_alto.py`

```python
"""
Data loader for ALTO XML + JPG page images.
Uses kraken's parsers for XML and line extraction.
"""
from PIL import Image
from kraken.lib.xml import XMLPage
from kraken.lib.segmentation import extract_polygons
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
```

#### Classe `AltoLineDataset(Dataset)`

Responsabilites :
- Scanner un repertoire pour trouver les paires .xml/.jpg
- Pour chaque page, parser l'ALTO avec `XMLPage(xml_path, filetype='alto')`
- Extraire les lignes avec `extract_polygons(pil_image, segmentation)`
- Stocker en memoire : liste de tuples `(line_image_array, text, source_page)`
- Normaliser chaque image de ligne a une hauteur fixe (ex: 48px), largeur
  variable proportionnelle
- `__getitem__` retourne `(img_tensor, text_string)`

```python
class AltoLineDataset(Dataset):
    def __init__(self, alto_dirs, img_height=48, max_width=2000):
        """
        Args:
            alto_dirs: list of directories containing .xml/.jpg pairs
            img_height: target line height in pixels
            max_width: discard lines wider than this (aberrant segmentation)
        """
        self.samples = []  # list of (np.array H x W, str)
        
        for alto_dir in alto_dirs:
            xml_files = sorted(glob.glob(os.path.join(alto_dir, '*.xml')))
            for xml_path in xml_files:
                if os.path.basename(xml_path) == 'METS.xml':
                    continue  # skip METS manifest
                
                jpg_path = xml_path.replace('.xml', '.jpg')
                if not os.path.exists(jpg_path):
                    continue
                
                page = XMLPage(xml_path, filetype='alto')
                seg = page.to_container()
                pil_img = Image.open(jpg_path)
                
                for line_img, line_obj in extract_polygons(pil_img, seg):
                    text = line_obj.text
                    if not text or not text.strip():
                        continue
                    
                    # Resize to target height, keep aspect ratio
                    w, h = line_img.size
                    if h == 0:
                        continue
                    new_w = int(w * img_height / h)
                    if new_w > max_width or new_w < 10:
                        continue
                    
                    line_img = line_img.resize((new_w, img_height),
                                              Image.LANCZOS)
                    # Convert to grayscale numpy
                    arr = np.array(line_img.convert('L'), dtype=np.float32)
                    self.samples.append((arr, text))
        
        print(f"Loaded {len(self.samples)} lines from {len(alto_dirs)} dirs")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        arr, text = self.samples[idx]
        img = torch.from_numpy(arr) / 255.0  # (H, W) normalized
        return img, text
```

#### Fonction `collate_alto_fn`

Meme logique que l'existant mais adapte aux donnees reelles :
- Extraction de colonnes par sliding window
- Encodage des textes en indices CTC
- Retourne `(padded_img_seqs, encoded_targets, input_lengths, target_lengths)`

```python
def collate_alto_fn(batch, window_size=10, stride=5, char_to_idx=None):
    """
    Args:
        batch: list of (img_tensor H x W, text_string)
        char_to_idx: dict mapping characters to CTC indices (0 = blank)
    Returns:
        img_seqs: (B, T_max, H, W_window) padded
        targets: (sum of target_lengths,) concatenated CTC targets
        input_lengths: (B,) actual sequence lengths
        target_lengths: (B,) text lengths
    """
    from generate_data import extract_columns  # reuse existing function
    
    img_seqs = []
    all_targets = []
    input_lengths = []
    target_lengths = []
    
    for img, text in batch:
        cols = extract_columns(img, window_size=window_size, stride=stride)
        img_seqs.append(cols)
        input_lengths.append(cols.shape[0])
        
        # Encode text
        encoded = [char_to_idx.get(c, 0) for c in text if c in char_to_idx]
        all_targets.extend(encoded)
        target_lengths.append(len(encoded))
    
    # Pad sequences
    max_len = max(seq.shape[0] for seq in img_seqs)
    B = len(img_seqs)
    H = img_seqs[0].shape[1]
    
    padded = torch.zeros(B, max_len, H, window_size)
    for i, seq in enumerate(img_seqs):
        padded[i, :seq.shape[0]] = seq
    
    targets = torch.tensor(all_targets, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    
    return padded, targets, input_lengths, target_lengths
```

#### Construction de l'alphabet

Scanner toutes les transcriptions pour construire `char_to_idx` :

```python
def build_alphabet(alto_dirs):
    """Scan all ALTO files and collect unique characters."""
    chars = set()
    for alto_dir in alto_dirs:
        for xml_path in glob.glob(os.path.join(alto_dir, '*.xml')):
            if 'METS' in xml_path:
                continue
            page = XMLPage(xml_path, filetype='alto')
            for line in page.get_sorted_lines():
                if hasattr(line, 'text') and line.text:
                    chars.update(line.text)
    
    chars = sorted(chars)
    char_to_idx = {c: i+1 for i, c in enumerate(chars)}  # 0 = CTC blank
    idx_to_char = {i+1: c for i, c in enumerate(chars)}
    idx_to_char[0] = ''  # blank
    
    print(f"Alphabet: {len(chars)} characters")
    print(f"Characters: {''.join(chars)}")
    return char_to_idx, idx_to_char
```

### Verification phase 0

```bash
python -c "
from data_alto import AltoLineDataset, build_alphabet
dirs = [
    'D:/OCR_genealogie/Alto/bars_dordogne_alto',
    'D:/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_1',
]
alphabet, _ = build_alphabet(dirs)
ds = AltoLineDataset(dirs, img_height=48)
print(f'Lines: {len(ds)}')
img, text = ds[0]
print(f'Shape: {img.shape}, Text: {text[:50]}')
"
```

---

## Phase 1 : Config et encodage texte

**Objectif** : preparer la config pour le multi-tache.

### Fichier : `config.py`

Ajouter les constantes suivantes :

```python
# CTC
CTC_BLANK = 0
LAMBDA_CTC = 1.0

# Architecture v2
ENCODER_TYPE = 'conv2d'     # 'mlp' pour l'ancien
IMG_HEIGHT = 48             # augmente de 32 a 48 pour les donnees reelles
EMBEDDING_DIM = 96          # augmente de 64 a 96 pour plus de capacite

# Donnees reelles
ALTO_DIRS = [
    'D:/OCR_genealogie/Alto/bars_dordogne_alto',
    'D:/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_1',
    'D:/OCR_genealogie/Alto/saint_chamassy_dordogne_alto_set_train',
]

# Note: ALPHABET, NUM_CLASSES, CHAR_TO_IDX, IDX_TO_CHAR
# seront construits dynamiquement par build_alphabet() au premier run
# et sauvegardes dans un fichier alphabet.json
```

### Fichier : `generate_data.py` (modifications mineures)

Ajouter les fonctions utilitaires d'encodage/decodage texte.
Ces fonctions sont generiques et utilisees par les deux pipelines
(synthetique et ALTO).

```python
def encode_text(text, char_to_idx):
    """Encode text string to list of CTC label indices."""
    return [char_to_idx[c] for c in text if c in char_to_idx]

def decode_indices(indices, idx_to_char):
    """Decode CTC output indices to string (greedy, with collapse)."""
    result = []
    prev = None
    for idx in indices:
        if idx != 0 and idx != prev:  # skip blanks and repeats
            result.append(idx_to_char.get(idx, '?'))
        prev = idx
    return ''.join(result)
```

Modifier `collate_fn` pour retourner le nouveau format :
`(padded_seqs, encoded_targets, input_lengths, target_lengths)`

L'ancien format `(padded_seqs, texts, lengths)` n'est plus retourne.

### Verification phase 1

L'entrainement existant (`train_light.py`) devra etre ajuste pour deballer
le nouveau format de collate, mais l'architecture du modele reste inchangee.

---

## Phase 2 : Encodeur Conv2D

**Objectif** : remplacer le MLP par un vrai CNN qui preserve la structure
spatiale.

### Fichier : `encoder.py`

Garder l'ancien `CNNEncoder` renomme en `MLPEncoder` (pour reference).

Nouvelle classe :

```python
class Conv2DEncoder(nn.Module):
    """
    Conv2D encoder for handwriting image columns.
    Preserves spatial structure unlike the MLP encoder.
    
    Input: (B, H, W) where H=48, W=window_size
    Output: (B, embedding_dim)
    """
    def __init__(self, img_height=48, window_size=10, embedding_dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            # (B, 1, 48, 10)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (B, 32, 24, 5)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # (B, 64, 12, 2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 1)),
            # (B, 128, 4, 1)
        )
        self.fc = nn.Linear(128 * 4, embedding_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            # (B, H*W) -> pas supporte, lever une erreur
            raise ValueError("Conv2DEncoder requires (B, H, W) input")
        # (B, H, W) -> (B, 1, H, W)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

Avec H=48, W=10, D=96 :
- Conv layers : ~75K params
- FC : 512 * 96 = ~49K params
- Total encoder : ~124K params

Note : l'AdaptiveAvgPool2d(4,1) rend l'encodeur robuste aux variations
de taille de fenetre. Le meme encodeur fonctionnera avec window_size
different sans changer l'architecture.

### Fichier : `model.py`

Nouvelle classe `HWMv2` (garder `HWMv1` intact) :

```python
class HWMv2(nn.Module):
    def __init__(self, img_height=48, window_size=10, embedding_dim=96,
                 num_layers=2, num_heads=2, ff_dim=192, dropout=0.1,
                 num_classes=None):
        super().__init__()
        self.encoder = Conv2DEncoder(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout)
        
        if num_classes is not None:
            self.ctc_head = CTCHead(embedding_dim, num_classes)
        else:
            self.ctc_head = None
        
        self.criterion = HybridLoss(...)
```

L'interface `encode_sequence` reste identique : `(B,T,H,W) -> (B,T,D)`.

### Verification phase 2

```bash
python -c "
import torch
from encoder import Conv2DEncoder
enc = Conv2DEncoder(img_height=48, window_size=10, embedding_dim=96)
x = torch.randn(4, 48, 10)
z = enc(x)
print(f'Input: {x.shape} -> Output: {z.shape}')
params = sum(p.numel() for p in enc.parameters())
print(f'Parameters: {params:,}')
"
```

---

## Phase 3 : Masque causal dans le Predictor

**Objectif** : empecher le transformer de "tricher" en regardant les tokens
futurs lors de la prediction.

### Fichier : `predictor.py`

Modifications minimales dans `TransformerPredictor` :

```python
def _generate_causal_mask(self, seq_len, device):
    """Upper triangular mask: True = masked (cannot attend)."""
    return torch.triu(torch.ones(seq_len, seq_len, device=device),
                      diagonal=1).bool()

def forward(self, z_seq):
    z_seq = self.pos_encoder(z_seq)
    
    # Masque causal : position t ne voit que positions 0..t
    causal_mask = self._generate_causal_mask(z_seq.size(1), z_seq.device)
    encoded = self.transformer(z_seq, mask=causal_mask)
    
    z_last = encoded[:, -1, :]
    z_pred = self.output_proj(z_last)
    return z_pred
```

Aucun changement de parametres. Le `predict_sequence` (autoregressive)
fonctionne deja correctement.

### Verification phase 3

```bash
python -c "
import torch
from predictor import TransformerPredictor
pred = TransformerPredictor(embedding_dim=96, num_layers=2, num_heads=2,
                            ff_dim=192)
z = torch.randn(4, 20, 96)
out = pred(z)
print(f'Input: {z.shape} -> Prediction: {out.shape}')
"
```

---

## Phase 4 : Tete CTC + Loss hybride

### Nouveau fichier : `ctc_head.py`

```python
"""CTC Recognition Head - maps embeddings to character logits."""
import torch.nn as nn

class CTCHead(nn.Module):
    """
    Linear projection from embedding space to character probabilities.
    
    Input: (B, T, D) embedding sequence
    Output: (B, T, num_classes) log-probabilities
    """
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.proj = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, z_seq):
        return self.proj(z_seq).log_softmax(dim=-1)
```

~num_classes * embedding_dim params. Avec D=96, C=80 (estimation alphabet
reel) : ~7.7K params. Negligeable.

### Fichier : `loss.py` (ajouter `HybridLoss`)

```python
class HybridLoss(nn.Module):
    """
    Combined loss: prediction + SIGReg + CTC
    
    L = L_pred + lambda_sigreg * L_sigreg + lambda_ctc * L_ctc
    """
    def __init__(self, lambda_sigreg=0.1, lambda_ctc=1.0):
        super().__init__()
        self.pred_loss = nn.MSELoss()
        self.sigreg = SIGRegLoss(lambda_reg=1.0)
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean',
                                    zero_infinity=True)
        self.lambda_sigreg = lambda_sigreg
        self.lambda_ctc = lambda_ctc
    
    def forward(self, z_pred, z_target, z_all,
                ctc_logits=None, targets=None,
                input_lengths=None, target_lengths=None):
        """
        Args:
            z_pred: (B, D) predicted next embedding
            z_target: (B, D) actual next embedding
            z_all: (B, T, D) all embeddings (for SIGReg)
            ctc_logits: (B, T, C) log-probs from CTCHead (optional)
            targets: (sum(target_lengths),) concatenated CTC targets
            input_lengths: (B,) sequence lengths
            target_lengths: (B,) target text lengths
        """
        # Prediction loss (auto-supervised)
        pred = self.pred_loss(z_pred, z_target)
        
        # SIGReg (auto-supervised)
        sigreg = self.sigreg(z_all)
        
        total = pred + self.lambda_sigreg * sigreg
        losses = {'pred': pred.item(), 'sigreg': sigreg.item()}
        
        # CTC loss (supervised, optional)
        if ctc_logits is not None and targets is not None:
            # CTC expects (T, B, C) input
            ctc_input = ctc_logits.permute(1, 0, 2)
            ctc = self.ctc_loss(ctc_input, targets,
                                input_lengths, target_lengths)
            total = total + self.lambda_ctc * ctc
            losses['ctc'] = ctc.item()
        
        losses['total'] = total.item()
        return total, losses
```

Point important : quand `ctc_logits=None`, la loss se comporte exactement
comme l'ancienne `HWMLoss`. Ca permet l'entrainement auto-supervise seul
(adaptation a un nouveau scripteur sans labels).

### Fichier : `model.py` (completer `HWMv2`)

```python
class HWMv2(nn.Module):
    def __init__(self, ..., num_classes=None, lambda_sigreg=0.1,
                 lambda_ctc=1.0):
        super().__init__()
        self.encoder = Conv2DEncoder(img_height, window_size, embedding_dim)
        self.predictor = TransformerPredictor(
            embedding_dim, num_layers, num_heads, ff_dim, dropout)
        
        self.ctc_head = CTCHead(embedding_dim, num_classes) \
                        if num_classes else None
        self.criterion = HybridLoss(lambda_sigreg, lambda_ctc)
    
    def forward(self, img_columns):
        z_seq = self.encode_sequence(img_columns)    # (B, T, D)
        z_history = z_seq[:, :-1, :]                  # (B, T-1, D)
        z_pred = self.predictor(z_history)            # (B, D)
        
        ctc_logits = None
        if self.ctc_head is not None:
            ctc_logits = self.ctc_head(z_seq)         # (B, T, C)
        
        return z_pred, z_seq, ctc_logits
    
    def compute_loss(self, img_columns, targets=None,
                     input_lengths=None, target_lengths=None):
        z_pred, z_seq, ctc_logits = self.forward(img_columns)
        z_target = z_seq[:, -1, :]
        
        return self.criterion(z_pred, z_target, z_seq,
                              ctc_logits, targets,
                              input_lengths, target_lengths)
    
    def adapt(self, img_columns):
        """
        Auto-supervised adaptation only (no labels needed).
        For adapting to a new scriptor without transcriptions.
        """
        z_pred, z_seq, _ = self.forward(img_columns)
        z_target = z_seq[:, -1, :]
        # CTC head not used — only prediction + SIGReg
        return self.criterion(z_pred, z_target, z_seq)
```

La methode `adapt()` est le coeur de la proposition de valeur : elle permet
l'adaptation a un nouveau scripteur avec zero labels.

### Verification phase 4

```bash
python -c "
import torch
from model import HWMv2
model = HWMv2(img_height=48, window_size=10, embedding_dim=96,
              num_classes=80)
x = torch.randn(4, 20, 48, 10)
z_pred, z_seq, logits = model(x)
print(f'z_pred: {z_pred.shape}')
print(f'z_seq: {z_seq.shape}')
print(f'logits: {logits.shape}')
print(f'Params: {model.count_parameters():,}')

# Test loss
targets = torch.randint(1, 80, (40,))
input_lengths = torch.full((4,), 20)
target_lengths = torch.full((4,), 10)
loss, losses = model.compute_loss(x, targets, input_lengths, target_lengths)
print(f'Losses: {losses}')

# Test adapt (no labels)
loss_adapt, losses_adapt = model.adapt(x)
print(f'Adapt losses: {losses_adapt}')
"
```

---

## Phase 5 : Entrainement multi-tache

### Nouveau fichier : `train.py` (remplace `train_light.py`)

Garder `train_light.py` intact comme reference.

```python
"""
Multi-task training for HWM-v2.
Supports:
  - Supervised: prediction + SIGReg + CTC (with ALTO data)
  - Self-supervised: prediction + SIGReg only (adapt mode)
"""
```

#### Fonction `train_epoch`

```python
def train_epoch(model, loader, optimizer, device, epoch, mode='full'):
    """
    Args:
        mode: 'full' = multi-task (pred + sigreg + ctc)
              'adapt' = self-supervised only (pred + sigreg)
    """
    model.train()
    totals = defaultdict(float)
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        if mode == 'full':
            img_seqs, targets, input_lengths, target_lengths = batch
            img_seqs = img_seqs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            if img_seqs.shape[1] < 2:
                continue
            
            optimizer.zero_grad()
            loss, losses = model.compute_loss(
                img_seqs, targets, input_lengths, target_lengths)
        
        elif mode == 'adapt':
            img_seqs = batch[0].to(device)  # ignore labels
            if img_seqs.shape[1] < 2:
                continue
            
            optimizer.zero_grad()
            loss, losses = model.adapt(img_seqs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        for k, v in losses.items():
            totals[k] += v
        num_batches += 1
    
    return {k: v / num_batches for k, v in totals.items()}
```

#### Fonction `train` principale

```python
def train(model, train_loader, val_loader=None, num_epochs=30,
          lr=1e-3, device='cpu', mode='full', save_path='hwm_v2.pt'):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    for epoch in range(1, num_epochs + 1):
        losses = train_epoch(model, train_loader, optimizer, device,
                             epoch, mode=mode)
        scheduler.step()
        
        print(f"Epoch {epoch}/{num_epochs} - " +
              " | ".join(f"{k}={v:.4f}" for k, v in losses.items()))
        
        # Evaluation CTC si disponible
        if val_loader and model.ctc_head:
            cer = evaluate_cer(model, val_loader, device)
            print(f"  Val CER: {cer:.1%}")
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': { ... },
        'alphabet': char_to_idx,
    }, save_path)
```

#### Script CLI

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'adapt'], default='full')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data', default='alto',
                        choices=['alto', 'synthetic'])
    parser.add_argument('--alto-dirs', nargs='+', default=config.ALTO_DIRS)
    parser.add_argument('--checkpoint', default=None,
                        help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Build alphabet from data
    if args.data == 'alto':
        char_to_idx, idx_to_char = build_alphabet(args.alto_dirs)
        num_classes = len(char_to_idx) + 1  # +1 for blank
        dataset = AltoLineDataset(args.alto_dirs, img_height=config.IMG_HEIGHT)
        # Split 80/20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        collate = partial(collate_alto_fn, char_to_idx=char_to_idx)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                                collate_fn=collate)
    
    # Create model
    model = HWMv2(
        img_height=config.IMG_HEIGHT,
        window_size=config.WINDOW_SIZE,
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=num_classes if args.mode == 'full' else None,
    )
    
    train(model, train_loader, val_loader, num_epochs=args.epochs,
          lr=args.lr, mode=args.mode)
```

Usage :
```bash
# Entrainement complet (supervise + auto-supervise)
python train.py --mode full --data alto --epochs 30

# Adaptation a un nouveau scripteur (auto-supervise uniquement)
python train.py --mode adapt --data alto \
    --alto-dirs D:/new_scriptor_pages/ \
    --checkpoint hwm_v2.pt --epochs 5 --lr 1e-4
```

### Verification phase 5

- Les 3 losses (pred, sigreg, ctc) diminuent sur les donnees ALTO
- L'entrainement tourne sans OOM avec batch_size=8
- Le checkpoint se sauvegarde correctement

---

## Phase 6 : Decodage CTC + evaluation

### Fichier : `recognize.py` (refactorisation complete)

Remplacer le linear probe par une evaluation CTC integree.

#### Decodage greedy

```python
def ctc_greedy_decode(log_probs, lengths, idx_to_char):
    """
    Greedy CTC decoding.
    
    Args:
        log_probs: (B, T, C) log-probabilities
        lengths: (B,) actual sequence lengths
        idx_to_char: dict mapping indices to characters
    Returns:
        list of decoded strings
    """
    results = []
    preds = log_probs.argmax(dim=-1)  # (B, T)
    
    for i in range(preds.size(0)):
        seq = preds[i, :lengths[i]].tolist()
        # Collapse repeats and remove blanks
        decoded = []
        prev = None
        for idx in seq:
            if idx != 0 and idx != prev:
                decoded.append(idx_to_char.get(idx, '?'))
            prev = idx
        results.append(''.join(decoded))
    
    return results
```

#### Metriques

```python
def levenshtein(s1, s2):
    """Edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j+1]+1, curr[j]+1,
                           prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def compute_cer(predictions, ground_truths):
    """Character Error Rate = sum(edit_distances) / sum(gt_lengths)."""
    total_dist = 0
    total_len = 0
    for pred, gt in zip(predictions, ground_truths):
        total_dist += levenshtein(pred, gt)
        total_len += len(gt)
    return total_dist / max(total_len, 1)
```

#### Evaluation integree

```python
def evaluate_cer(model, loader, device, idx_to_char):
    """Run full CTC evaluation on a DataLoader."""
    model.eval()
    all_preds = []
    all_gts = []
    
    with torch.no_grad():
        for img_seqs, targets, input_lengths, target_lengths in loader:
            img_seqs = img_seqs.to(device)
            _, z_seq, ctc_logits = model(img_seqs)
            
            decoded = ctc_greedy_decode(ctc_logits, input_lengths,
                                        idx_to_char)
            all_preds.extend(decoded)
            
            # Reconstruct ground truth strings
            offset = 0
            for tlen in target_lengths:
                gt_indices = targets[offset:offset+tlen].tolist()
                gt_text = ''.join(idx_to_char.get(i, '?')
                                  for i in gt_indices)
                all_gts.append(gt_text)
                offset += tlen
    
    cer = compute_cer(all_preds, all_gts)
    
    # Show examples
    print("\nExamples (first 10):")
    for pred, gt in zip(all_preds[:10], all_gts[:10]):
        mark = 'OK' if pred == gt else 'ERR'
        print(f"  [{mark}] GT:   {gt}")
        print(f"        PRED: {pred}")
    
    return cer
```

### Verification phase 6

```bash
python -c "
from recognize import evaluate_cer
# ... load model, create val_loader ...
cer = evaluate_cer(model, val_loader, 'cpu', idx_to_char)
print(f'CER: {cer:.1%}')
"
```

Objectifs :
- CER < 50% = l'architecture fonctionne
- CER < 20% = resultats exploitables
- CER < 10% = comparable a Kraken fine-tune

---

## Budget parametres final (D=96)

| Composant | Params |
|-----------|--------|
| Conv2D Encoder (3 couches) | ~124K |
| Transformer Predictor (2L, 2H, ff=192) | ~160K |
| CTC Head (~80 classes) | ~7.7K |
| **Total** | **~292K** (bien sous 1M) |

Marge pour augmenter : D=128, 4 couches transformer = ~600K, toujours
sous la limite.

---

## Resume des fichiers

| Fichier | Action | Phase |
|---------|--------|-------|
| `config.py` | Modifier | 1 |
| `data_alto.py` | **Creer** | 0 |
| `generate_data.py` | Modifier (ajout encode/decode) | 1 |
| `encoder.py` | Modifier (ajout Conv2DEncoder) | 2 |
| `predictor.py` | Modifier (masque causal) | 3 |
| `ctc_head.py` | **Creer** | 4 |
| `loss.py` | Modifier (ajout HybridLoss) | 4 |
| `model.py` | Modifier (ajout HWMv2) | 2, 4 |
| `train.py` | **Creer** | 5 |
| `recognize.py` | Modifier (CTC decode + CER) | 6 |

Fichiers NON modifies (archives PoC) :
- `train_light.py` — conserve tel quel
- `inference.py` — conserve tel quel
- `export_model.py` — a adapter plus tard si necessaire

---

## Ordre d'implementation recommande

Chaque phase est independamment testable. Un commit par phase.

```
Phase 0 : data_alto.py            -> commit "feat: ALTO data loader using kraken parsers"
Phase 1 : config.py, generate_data.py -> commit "feat: CTC text encoding and config v2"
Phase 2 : encoder.py, model.py    -> commit "feat: Conv2D encoder replacing MLP"
Phase 3 : predictor.py            -> commit "fix: add causal mask to transformer predictor"
Phase 4 : ctc_head.py, loss.py, model.py -> commit "feat: CTC head and hybrid loss"
Phase 5 : train.py                -> commit "feat: multi-task training loop with adapt mode"
Phase 6 : recognize.py            -> commit "feat: CTC decoding and CER evaluation"
```

---

## Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| kraken ne s'installe pas proprement sur Windows | Bloquant phase 0 | Fallback: ecrire un parser ALTO minimal avec lxml |
| CTC loss NaN avec sequences courtes | Training instable | `zero_infinity=True` + filtre T >= 2*L+1 |
| Gradient CTC domine la prediction | Representations degradees | Ajuster lambda_ctc (0.1 -> 1.0), monitorer les 3 losses |
| Alphabet trop grand (accents, symboles) | CTC plus dur a apprendre | Normaliser le texte (unidecode), reduire l'alphabet |
| OOM sur les pages larges | Crash | Limiter max_width, batch_size adaptatif |
| L'adaptation auto-supervisee ne transfere pas | Valeur limitee | Evaluer CER avant/apres adapt sur scripteur inconnu |
