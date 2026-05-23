# État de l'Art de la Reconnaissance d'Écriture Manuscrite (HTR) - 2026

## Rapport de Recherche : HTR, Écriture Ancienne et Positionnement de l'Approche Scribe World Model

**Date :** Avril 2026  
**Version :** 1.0

---

## Table des Matières

1. [Introduction](#1-introduction)
2. [État de l'Art HTR](#2-état-de-lart-htr)
3. [Focus Écriture Ancienne](#3-focus-écriture-ancienne)
4. [Analyse de l'Approche Scribe World Model](#4-analyse-de-lapproche-scribe-world-model)
5. [Benchmarks et Objectifs de Performance](#5-benchmarks-et-objectifs-de-performance)
6. [Outils et Frameworks Disponibles](#6-outils-et-frameworks-disponibles)
7. [Conclusion et Recommandations](#7-conclusion-et-recommandations)

---

## 1. Introduction

### 1.1 Contexte

La reconnaissance d'écriture manuscrite (HTR - Handwritten Text Recognition) est un domaine de recherche en pleine évolution, particulièrement pertinent pour la numérisation du patrimoine culturel et l'exploitation des archives historiques. Ce rapport présente une analyse approfondie de l'état de l'art en 2026, avec un focus particulier sur :

- Les architectures dominantes et leurs performances
- Les défis spécifiques de l'écriture ancienne (manuscrits historiques, archives)
- Le positionnement d'une approche innovante : le "Scribe World Model"
- Les outils et frameworks open-source disponibles

### 1.2 Périmètre

Cette étude couvre :
- Les publications scientifiques récentes (2023-2026)
- Les datasets de référence (IAM, RIMES, Bentham, Washington, etc.)
- Les métriques de performance (CER, WER)
- Les outils open-source (Kraken, Calamari, Transkribus)

---

## 2. État de l'Art HTR

### 2.1 Architectures Dominantes

#### 2.1.1 Vision Transformers (ViT) pour HTR

L'architecture **HTR-VT** (2024) représente une avancée majeure dans l'application des Vision Transformers à la reconnaissance d'écriture manuscrite. Cette approche surpasse les méthodes traditionnelles CNN-LSTM sur plusieurs benchmarks.

**Caractéristiques clés :**
- Utilisation de patches d'image comme tokens d'entrée
- Mécanisme d'attention multi-head pour capturer les dépendances globales
- Intégration de position embeddings pour préserver la structure spatiale

#### 2.1.2 Approches Hybrides CNN-Transformer

**HTR-ConvText** (décembre 2025) introduit une architecture hybride combinant :
- Couches convolutives pour l'extraction de features locales (détails de traits, courbures)
- Encodeur Transformer pour la modélisation des dépendances séquentielles à long terme
- Décodage CTC ou attention-based

**Performances :**
- Réduction du CER de 15-20% par rapport aux architectures CNN-LSTM pures
- Meilleure généralisation sur les styles d'écriture non vus

#### 2.1.3 TrOCR (Microsoft)

**TrOCR** (Transformer-based Optical Character Recognition) est un modèle pré-entraîné par Microsoft qui a établi de nouveaux standards :

- Architecture encodeur-décodeur Transformer
- Pré-entraînement sur de grandes quantités de données synthétiques et réelles
- Fine-tuning possible sur des datasets spécifiques

**Points forts :**
- Performances état de l'art sur IAM, RIMES et datasets historiques
- Modèles disponibles en open-source (Hugging Face)
- Facilité de fine-tuning

#### 2.1.4 DRetHTR - Decoder-only RetNet

Architecture innovante basée sur RetNet (alternative aux Transformers avec complexité linéaire) :

**Performances records :**
- IAM-A (anglais) : **2.26% CER**
- RIMES (français) : **1.81% CER**
- Bentham (historique anglais) : **3.46% CER**
- READ-2016 (allemand historique) : **4.21% CER**

#### 2.1.5 HTR-JAND (2024)

Architecture avec Joint Attention Network et Knowledge Distillation :

**Performances exceptionnelles :**
- IAM : **1.23% CER**
- RIMES : **1.02% CER**
- Bentham : **2.02% CER**

La réduction de 62.41% du CER grâce au Knowledge Distillation seul (de 12.21% à 4.59% sur IAM) démontre l'importance de cette technique.

### 2.2 Datasets de Référence

#### 2.2.1 Datasets Modernes

| Dataset | Langue | Taille | Caractéristiques |
|---------|--------|--------|------------------|
| **IAM** | Anglais | ~115K mots | Écriture moderne, lignes de texte |
| **RIMES** | Français | ~350K mots | Courrier administratif moderne |
| **CVL** | Multilingue | ~82K mots | Base de données collaborative |
| **IAM-A** | Anglais | Variante augmentée | Version augmentée d'IAM |

#### 2.2.2 Datasets Historiques

| Dataset | Période | Langue | Caractéristiques |
|---------|---------|--------|------------------|
| **Bentham** | XVIIIe-XIXe s. | Anglais | Manuscrits philosophiques |
| **Washington** | XVIIIe s. | Anglais | Lettres de George Washington |
| **READ-2016** | XVe-XVIe s. | Allemand | Documents historiques allemands |
| **Saint Gall** | IXe s. | Latin | Manuscrits médiévaux |
| **Rodrigo** | XVIe s. | Espagnol | Documents historiques espagnols |

### 2.3 Métriques de Performance

#### 2.3.1 Character Error Rate (CER)

Le CER est la métrique standard pour HTR :

```
CER = (S + D + I) / N
```

Où :
- S = substitutions
- D = deletions
- I = insertions
- N = nombre total de caractères dans la ground truth

#### 2.3.2 Word Error Rate (WER)

```
WER = (S + D + I) / N
```

Calculé au niveau du mot plutôt que du caractère.

#### 2.3.3 Performances État de l'Art 2024-2026

| Dataset | Meilleur CER | Modèle | Année |
|---------|--------------|--------|-------|
| IAM | 1.23% | HTR-JAND | 2024 |
| IAM-A | 2.26% | DRetHTR | 2024 |
| RIMES | 1.02% | HTR-JAND | 2024 |
| Bentham | 2.02% | HTR-JAND | 2024 |
| READ-2016 | 4.21% | DRetHTR | 2024 |

---

## 3. Focus Écriture Ancienne

### 3.1 Défis Spécifiques

#### 3.1.1 Variabilité de l'Écriture

L'écriture ancienne présente des défis uniques :

1. **Variabilité inter-scribe** : Différences significatives entre les styles des différents auteurs
2. **Évolution temporelle** : Changement des styles d'écriture au fil des siècles
3. **Absence de standardisation** : Orthographe variable, abréviations, ligatures

#### 3.1.2 Qualité des Documents

- Dégradation physique (taches, déchirures, papier jauni)
- Variation de l'encre et du contraste
- Show-through (transparence du verso)
- Marginalia et annotations superposées

#### 3.1.3 Défis Linguistiques

- Vocabulaire archaïque et obsolète
- Variation orthographique (ex: "françoys" vs "français")
- Latin médiéval avec variations régionales
- Abréviations systématiques non standardisées

### 3.2 Approches pour Documents Historiques

#### 3.2.1 LLMs pour la Transcription de Documents Historiques

Une étude majeure de 2024 a évalué l'utilisation des LLMs pour améliorer les transcriptions HTR de documents historiques :

**Méthodologie :**
- Utilisation de LLMs (GPT-4, Claude, modèles open-source) pour la correction post-OCR
- Expérimentation sur documents du XIXe siècle britannique
- Évaluation sur données de test et annotations manuelles

**Résultats :**
- Correction moyenne de 54% des erreurs par rapport au décodage standard
- Amélioration significative de la lisibilité des transcriptions
- Potentiel pour réduire le travail manuel de correction

#### 3.2.2 Adaptation au Domaine

**Fine-tuning sur données historiques :**
- Entraînement sur des datasets spécifiques à la période
- Utilisation de lexiques historiques pour le décodage
- Augmentation de données simulant les dégradations

**Transfer Learning :**
- Pré-entraînement sur données modernes abondantes
- Fine-tuning sur le domaine historique cible
- Approches multi-tâches pour améliorer la généralisation

### 3.3 Performances sur Documents Historiques

#### 3.3.1 Benchmark READ-2016

Le dataset READ-2016 (documents allemands du XVe-XVIe siècle) est particulièrement difficile :

- **Meilleur CER :** 4.21% (DRetHTR)
- **CER moyen état de l'art :** 5-8%
- **CER baseline CNN-LSTM :** 10-15%

#### 3.3.2 Dataset Bentham

Documents philosophiques de Jeremy Bentham (XVIIIe-XIXe s.) :

- **Meilleur CER :** 2.02% (HTR-JAND)
- **CER TrOCR :** ~3-4%
- **CER baseline :** 8-12%

#### 3.3.3 Écart de Performance Moderne vs Historique

| Type | CER État de l'Art | Difficulté Relative |
|------|-------------------|---------------------|
| Moderne (IAM) | 1.23% | Baseline |
| Historique simple (Bentham) | 2.02% | +64% |
| Historique complexe (READ-2016) | 4.21% | +242% |

### 3.4 Génération d'Écriture Manuscrite (Handwriting Synthesis)

#### 3.4.1 État de l'Art

Un survey complet (2019-2024) couvre les avancées en génération d'écriture :

**Approches dominantes :**
- **GAN-based** : Style transfer et génération conditionnée
- **Transformer-based** : Modélisation séquentielle des traits
- **Diffusion models** : Génération haute qualité (émergent en 2024-2025)

#### 3.4.2 Applications pour HTR

- **Data Augmentation** : Génération de données d'entraînement synthétiques
- **Style Normalization** : Transformation vers un style canonique
- **On-to-Off Transformation** : Conversion écriture en ligne → hors ligne

#### 3.4.3 ScriptViT (2025)

Vision Transformer pour génération d'écriture personnalisée :

- Capture du style d'un écrivain spécifique
- Génération réaliste alignée avec le style cible
- Combinaison GAN + Transformer + Diffusion

---

## 4. Analyse de l'Approche Scribe World Model

### 4.1 Concept de World Model

#### 4.1.1 Définition

Un **World Model** est un modèle qui apprend une représentation interne du monde permettant de :
1. **Comprendre l'état présent** : Construire des représentations implicites du monde
2. **Prédire les états futurs** : Simuler l'évolution temporelle

#### 4.1.2 Origines et Fondements

Le concept provient de plusieurs domaines :

**Psychologie Cognitive :**
- Théorie des "mental models" (Kenneth Craik, 1943)
- Les humains construisent des modèles internes pour prédire et comprendre

**Reinforcement Learning :**
- Ha & Schmidhuber (2018) : "World Models" pour le RL
- Dreamer (DeepMind) : Apprentissage de représentations latentes

**Vision de LeCun (2022) :**
- JEPA (Joint Embedding Predictive Architecture)
- Modèle du monde pour l'intelligence artificielle générale

#### 4.1.3 World Models Modernes

**Applications actuelles :**
- **Sora** (OpenAI) : Génération vidéo avec cohérence temporelle
- **Cosmos** : Adhérence aux lois physiques
- **DreamerV3** : Contrôle robotique via modèles du monde appris

### 4.2 Application aux Strokes d'Écriture

#### 4.2.1 Modélisation des Strokes

L'écriture manuscrite peut être vue comme une **séquence temporelle de traits (strokes)** :

```
Stroke = [(x1, y1, t1), (x2, y2, t2), ..., (xn, yn, tn)]
```

Un World Model pour l'écriture pourrait :
1. **Apprendre la dynamique des traits** : Comment un trait évolue naturellement
2. **Prédire la suite du trait** : Anticiper le mouvement du stylo
3. **Modéliser le style** : Capturer les caractéristiques d'un écrivain

#### 4.2.2 Approche Predictive

**Contrastive Predictive Coding (CPC) :**
- Apprentissage de représentations qui maximisent l'information prédictive
- Discrimination entre états futurs vrais et négatifs
- Améliore l'abstraction temporelle

**Transformer-based World Models (TWISTER) :**
- Utilisation de Transformers pour la prédiction à long terme
- Codage prédictif contrastif action-conditionné
- Représentations temporelles de haut niveau

### 4.3 Originalité de l'Approche Scribe World Model

#### 4.3.1 Positionnement par Rapport à l'État de l'Art

| Aspect | HTR Traditionnel | Scribe World Model (hypothèse) |
|--------|------------------|--------------------------------|
| Paradigme | Reconnaissance image → texte | Modélisation prédictive des strokes |
| Architecture | CNN-LSTM ou Transformer encodeur | World model + prédiction séquentielle |
| Données | Images hors ligne | Trajectoires de strokes |
| Apprentissage | Supervisé (image, transcription) | Self-supervisé + supervisé |

#### 4.3.2 Points Forts Potentiels

1. **Modélisation explicite de la dynamique** :
   - Compréhension du processus d'écriture
   - Meilleure généralisation aux variations de style

2. **Apprentissage self-supervisé** :
   - Exploitation de grandes quantités de données non annotées
   - Réduction du besoin en ground truth

3. **Prédiction anticipative** :
   - Utilisation du contexte futur (si disponible)
   - Correction d'ambiguïtés via prédiction

4. **Cadre unifié** :
   - Reconnaissance + Génération dans un même modèle
   - Potentiel pour la synthesis d'écriture

#### 4.3.3 Limites et Défis

1. **Disponibilité des données de strokes** :
   - Les datasets classiques (IAM, RIMES) sont "offline" (images uniquement)
   - Nécessite des données "online" avec trajectoires temporelles

2. **Complexité computationnelle** :
   - Modélisation temporelle fine peut être coûteuse
   - Besoin d'inférence temps réel pour applications pratiques

3. **Évaluation** :
   - Métriques standards (CER, WER) ne capturent pas la qualité du world model
   - Besoin de nouvelles métriques pour évaluer la prédiction

4. **Écart avec les performances actuelles** :
   - HTR-JAND atteint 1.02% CER sur RIMES
   - Seuil de compétitivité très élevé

### 4.4 Comparaison avec les Méthodes Existantes

#### 4.4.1 vs CNN-LSTM (Baseline)

**CNN-LSTM :**
- Architecture mature, bien comprise
- Performances solides sur datasets modernes
- Faible capacité de modélisation du contexte global

**Scribe World Model :**
- Approche plus expressive
- Potentiel de meilleure généralisation
- Complexité accrue

#### 4.4.2 vs Transformers (TrOCR, HTR-VT)

**Transformers HTR :**
- État de l'art actuel
- Attention capture les dépendances globales
- Pré-entraînement sur grandes quantités de données

**Scribe World Model :**
- Focus sur la dynamique temporelle des strokes
- Potentiellement complémentaire aux Transformers
- Pourrait améliorer la robustesse aux variations de style

#### 4.4.3 vs Approches Hybrides (HTR-ConvText, HTR-JAND)

**Hybrides :**
- Combinaison optimale des forces CNN + Transformer
- Techniques avancées (Knowledge Distillation, Curriculum Learning)
- Performances records actuelles

**Scribe World Model :**
- Paradigme différent (modélisation prédictive)
- Pourrait être combiné avec les architectures hybrides
- Niche potentielle : écriture très variable / historique

---

## 5. Benchmarks et Objectifs de Performance

### 5.1 Métriques de Référence

#### 5.1.1 CER (Character Error Rate)

| Niveau | CER | Qualité |
|--------|-----|---------|
| Excellent | < 2% | Utilisable sans correction |
| Très bon | 2-5% | Correction mineure nécessaire |
| Bon | 5-10% | Correction significative |
| Acceptable | 10-15% | Nécessite révision importante |
| Faible | > 15% | Non utilisable en l'état |

#### 5.1.2 WER (Word Error Rate)

Le WER est généralement 2-3x supérieur au CER :

| Dataset | CER État de l'Art | WER Estimé |
|---------|-------------------|------------|
| IAM | 1.23% | ~3-4% |
| RIMES | 1.02% | ~2.5-3.5% |
| Bentham | 2.02% | ~5-6% |

### 5.2 Seuils de Compétitivité

#### 5.2.1 Pour Datasets Modernes

Pour être **compétitif** sur les datasets modernes (IAM, RIMES) :

- **Minimum acceptable :** CER < 5%
- **Compétitif :** CER < 3%
- **État de l'art :** CER < 2%

#### 5.2.2 Pour Documents Historiques

Pour les documents historiques (Bentham, READ-2016) :

- **Minimum acceptable :** CER < 10%
- **Compétitif :** CER < 6%
- **État de l'art :** CER < 4%

#### 5.2.3 Pour Archives Non Standardisées

Pour des archives très anciennes ou très dégradées :

- **Minimum acceptable :** CER < 20%
- **Compétitif :** CER < 12%
- **Bon :** CER < 8%

### 5.3 Benchmarks de Référence

#### 5.3.1 Modernes

| Modèle | IAM CER | RIMES CER | Architecture |
|--------|---------|-----------|--------------|
| HTR-JAND | 1.23% | 1.02% | Transformer + KD |
| DRetHTR | 2.26% | 1.81% | RetNet decoder-only |
| TrOCR Large | ~2-3% | ~2-3% | Transformer encoder-decoder |
| Baseline CNN-LSTM | 8-12% | 6-10% | CNN-BLSTM-CTC |

#### 5.3.2 Historiques

| Modèle | Bentham CER | READ-2016 CER | Notes |
|--------|-------------|---------------|-------|
| HTR-JAND | 2.02% | - | Fine-tuned |
| DRetHTR | 3.46% | 4.21% | Multi-dataset |
| TrOCR | ~3-4% | ~5-7% | Fine-tuned |
| Baseline | 8-12% | 10-15% | Sans adaptation |

### 5.4 Objectifs de Performance pour Scribe World Model

#### 5.4.1 Scénario Optimiste

Si l'approche World Model apporte une valeur significative :

| Dataset | Objectif CER | vs État de l'Art |
|---------|--------------|------------------|
| IAM | 2-3% | Compétitif |
| RIMES | 2-3% | Compétitif |
| Bentham | 3-4% | Compétitif |
| READ-2016 | 4-6% | Compétitif |

#### 5.4.2 Scénario Réaliste

Performances compétitives mais inférieures à l'état de l'art :

| Dataset | Objectif CER | vs État de l'Art |
|---------|--------------|------------------|
| IAM | 3-5% | Bon |
| RIMES | 3-5% | Bon |
| Bentham | 4-6% | Bon |
| READ-2016 | 6-8% | Bon |

#### 5.4.3 Scénario Niche

Focus sur les cas difficiles où le World Model excelle :

- **Styles très variables :** Meilleure généralisation que les approches standard
- **Peu de données d'entraînement :** Apprentissage self-supervisé
- **Reconnaissance + Génération :** Cadre unifié

---

## 6. Outils et Frameworks Disponibles

### 6.1 Kraken

#### 6.1.1 Description

**Kraken** est un système OCR/HTR open-source développé par l'Université de Liège, particulièrement adapté aux documents historiques.

**Caractéristiques :**
- Architecture : CNN-BLSTM-CTC
- Support natif des scripts non latins
- Outils de segmentation de ligne
- Pré-entraîné sur plusieurs datasets historiques

#### 6.1.2 Performances

| Dataset | CER Kraken | CER État de l'Art |
|---------|------------|-------------------|
| IAM | 5-8% | 1.23% |
| Bentham | 6-10% | 2.02% |
| READ-2016 | 8-12% | 4.21% |

#### 6.1.3 Avantages et Inconvénients

**Avantages :**
- Open-source, bien documenté
- Communauté active
- Spécialisé documents historiques
- Léger et rapide

**Inconvénients :**
- Performances inférieures aux Transformers récents
- Architecture plus ancienne (CNN-LSTM)
- Moins de modèles pré-entraînés

### 6.2 Calamari

#### 6.2.1 Description

**Calamari** est un framework HTR open-source avec support multi-modèles et techniques avancées.

**Caractéristiques :**
- Architecture : CNN-BLSTM-CTC
- Support du voting multi-modèles
- Cross-fold training pour la robustesse
- Intégration facile avec d'autres outils

#### 6.2.2 Performances

Similaires à Kraken sur les mêmes architectures, avec amélioration via le voting :

| Technique | Amélioration CER |
|-----------|------------------|
| Single model | Baseline |
| 5-model voting | -10 à -20% CER |
| 10-model voting | -15 à -25% CER |

#### 6.2.3 Points Forts

- Excellent pour la robustesse via ensemble
- Facilement extensible
- Bien intégré avec Transkribus

### 6.3 Transkribus

#### 6.3.1 Description

**Transkribus** est une plateforme commerciale avec composants open-source pour la reconnaissance de documents historiques.

**Caractéristiques :**
- Interface graphique complète
- Modèles PyLaia (Transformer-based)
- Collaboration et annotation
- Services cloud + option self-hosted

#### 6.3.2 PyLaia

Le moteur HTR de Transkribus :

- Architecture : CNN-BLSTM avec attention
- Modèles pré-entraînés sur plusieurs langues
- Fine-tuning via interface web
- Performances proches de l'état de l'art

#### 6.3.3 Utilisation

**Pour les institutions :**
- Solution clé en main
- Support et formation
- Infrastructure cloud

**Pour les chercheurs :**
- PyLaia disponible en open-source
- Possibilité de fine-tuning avancé
- Accès aux modèles pré-entraînés

### 6.4 TrOCR (Microsoft)

#### 6.4.1 Description

**TrOCR** est un modèle Transformer pré-entraîné par Microsoft, disponible via Hugging Face.

**Caractéristiques :**
- Architecture : Encoder-Decoder Transformer
- Pré-entraînement sur données synthétiques massives
- Modèles de différentes tailles (small, base, large)
- Fine-tuning sur datasets personnalisés

#### 6.4.2 Modèles Disponibles

| Modèle | Paramètres | Usage |
|--------|------------|-------|
| TrOCR-small | ~60M | Ressources limitées |
| TrOCR-base | ~330M | Usage général |
| TrOCR-large | ~550M | Meilleures performances |
| TrOCR-stage1 | - | Pré-entraînement |
| TrOCR-stage2 | - | Fine-tuning |

#### 6.4.3 Utilisation

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

# Inference
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

#### 6.4.4 Performances

| Dataset | CER TrOCR-large | CER État de l'Art |
|---------|-----------------|-------------------|
| IAM | ~2-3% | 1.23% |
| RIMES | ~2-3% | 1.02% |
| Bentham | ~3-4% | 2.02% |

### 6.5 Comparatif Global

| Outil | Type | Architecture | Facilité | Performance | Historique |
|-------|------|--------------|----------|-------------|------------|
| Kraken | Open-source | CNN-LSTM | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Calamari | Open-source | CNN-LSTM | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Transkribus | Commercial | PyLaia | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| TrOCR | Open-source | Transformer | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| HTR-JAND | Recherche | Transformer+KD | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### 6.6 Recommandations par Usage

#### 6.6.1 Pour la Recherche

- **TrOCR** : Baseline solide, facile à fine-tuner
- **HTR-JAND** : État de l'art (si code disponible)
- **PyLaia** : Bon compromis performance/contrôle

#### 6.6.2 Pour les Archives Historiques

- **Transkribus** : Solution complète institutionnelle
- **Kraken** : Alternative open-source légère
- **TrOCR + fine-tuning** : Bonnes performances avec effort

#### 6.6.3 Pour Prototypage Rapide

- **TrOCR pré-entraîné** : Zéro entraînement requis
- **Transkribus Lite** : Interface web simple
- **Kraken modèles pré-entraînés** : Bon point de départ

---

## 7. Conclusion et Recommandations

### 7.1 Synthèse de l'État de l'Art

#### 7.1.1 Architectures Gagnantes

Les **Transformers** (particulièrement avec Knowledge Distillation et techniques avancées) dominent actuellement :

- **HTR-JAND** : Performances records sur IAM (1.23%), RIMES (1.02%), Bentham (2.02%)
- **TrOCR** : Modèle pré-entraîné accessible, performances excellentes
- **DRetHTR** : Alternative RetNet avec complexité linéaire

#### 7.1.2 Datasets

- **Modernes** : IAM et RIMES sont les standards, avec CER < 2% atteignable
- **Historiques** : Bentham et READ-2016 représentent les défis actuels, CER 2-5%

#### 7.1.3 Tendances 2024-2026

1. **Pré-entraînement massif** : Modèles généraux fine-tunés sur le domaine
2. **Knowledge Distillation** : Compression et amélioration des performances
3. **Approches hybrides** : Combinaison CNN + Transformer
4. **LLMs pour post-correction** : Amélioration des transcriptions HTR

### 7.2 Analyse de l'Approche Scribe World Model

#### 7.2.1 Potentiel

L'approche World Model appliquée à l'écriture présente un **potentiel d'innovation** significatif :

1. **Paradigme original** : Modélisation prédictive vs reconnaissance pure
2. **Cadre unifié** : Reconnaissance + Génération
3. **Self-supervision** : Exploitation de données non annotées

#### 7.2.2 Défis

1. **Disponibilité des données de strokes** : Besoin de données temporelles (online)
2. **Seuil de compétitivité élevé** : État de l'art déjà très performant (CER < 2%)
3. **Validation empirique** : Nécessite implémentation et benchmarks

#### 7.2.3 Positionnement Recommandé

**Niche cible :**
- Documents historiques à **forte variabilité** de style
- Contextes avec **peu de données d'entraînement**
- Applications nécessitant **génération et reconnaissance** unifiées

**Pas recommandé pour :**
- Datasets modernes (IAM, RIMES) où les Transformers excellent
- Applications nécessitant CER < 2% immédiatement

### 7.3 Recommandations

#### 7.3.1 Pour Développer Scribe World Model

1. **Commencer par un prototype** :
   - Implémenter un world model simple pour les strokes
   - Évaluer sur un sous-ensemble de données online

2. **Identifier les cas d'usage différenciants** :
   - Documents avec forte variabilité inter-scribe
   - Styles d'écriture non vus à l'entraînement
   - Génération d'écriture synthétique pour data augmentation

3. **Combiner avec l'état de l'art** :
   - World Model comme composant dans une architecture hybride
   - Utiliser TrOCR ou PyLaia comme backbone
   - Ajouter la modélisation prédictive comme amélioration

4. **Objectifs de performance réalistes** :
   - CER < 5% sur datasets modernes (compétitif)
   - CER < 8% sur datasets historiques (bon)
   - Focus sur la **généralisation aux styles non vus**

#### 7.3.2 Pour les Projets HTR Pratiques

1. **Débuter avec TrOCR** :
   - Utiliser les modèles pré-entraînés
   - Fine-tuner sur le domaine cible
   - Attendre CER 3-5% sans effort majeur

2. **Pour les documents historiques** :
   - Évaluer Transkribus (version gratuite)
   - Fine-tuner TrOCR sur des données annotées
   - Considérer Kraken pour des ressources limitées

3. **Post-traitement avec LLMs** :
   - Utiliser GPT-4 ou Claude pour la correction
   - Amélioration potentielle de 50%+ des erreurs
   - Évaluer le coût vs bénéfice

#### 7.3.3 Pour les Archives Françaises

Les archives françaises présentent des défis spécifiques :

1. **Datasets disponibles** :
   - RIMES pour l'écriture moderne française
   - Datasets spécifiques pour les périodes historiques (XVIIIe-XIXe s.)

2. **Recommandations** :
   - Fine-tuner TrOCR sur des données d'époque
   - Utiliser des lexiques historiques pour le décodage
   - Combiner avec des modèles de langue français anciens

### 7.4 Pistes de Recherche Futures

1. **World Models pour HTR** :
   - Évaluation systématique sur benchmarks standard
   - Intégration dans architectures hybrides
   - Applications à la génération de données synthétiques

2. **LLMs + HTR** :
   - Post-correction systématique
   - Modèles de langue historiques
   - Annotation semi-automatique

3. **Approches multi-modales** :
   - Combinaison texte + image + métadonnées
   - Utilisation du contexte documentaire
   - Modèles foundation pour les archives

---

## Annexe A : Références

### Papers Clés 2024-2026

1. **HTR-JAND** (décembre 2024) : "Handwritten Text Recognition with Joint Attention Network and Knowledge Distillation" - arXiv:2412.18524

2. **HTR-VT** (septembre 2024) : "A Vision Transformer-based Approach for Handwritten Text Recognition" - arXiv:2409.08573

3. **HTR-ConvText** (décembre 2025) : "Hybrid CNN-Transformer for Handwritten Text Recognition"

4. **DRetHTR** (2024) : "Decoder-only RetNet for Handwritten Text Recognition"

5. **TrOCR** (2021-2024) : "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models" - Microsoft

6. **World Models Survey** (novembre 2025) : "Understanding World or Predicting Future? A Comprehensive Survey of World Models" - arXiv:2411.14499

7. **LLMs for Historical Documents** (2024) : Évaluation des LLMs pour la transcription de documents historiques - PMC

8. **Handwriting Synthesis Survey** (janvier 2025) : "A survey of handwriting synthesis from 2019 to 2024" - ScienceDirect

### Outils

- Kraken : https://kraken.re/
- Calamari : https://github.com/Calamari-OCR/calamari
- Transkribus : https://transkribus.eu/
- TrOCR : https://huggingface.co/docs/transformers/model_doc/trocr

---

**Fin du Rapport**

*Généré en avril 2026 - État de l'art HTR et positionnement Scribe World Model*
