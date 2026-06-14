"""Informations sur les checkpoints du modèle HWM.

Ce module fournit des fonctions pour lire les métadonnées des fichiers
de checkpoint (.pt) sans charger les poids du modèle, ce qui permet
d'afficher un résumé dans l'interface NiceGUI sans solliciter le GPU.
"""

import torch
import os
import glob
import json
import re
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Analyse du nom de fichier
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """Analyse un nom de fichier checkpoint pour en extraire la version et les tags.

    Convention de nommage :

    * ``hwm_v17.pt``           → version « v17 », aucun tag
    * ``hwm_v11_adapt.pt``     → version « v11 », tag « adapt »
    * ``hwm_v18_322_bars.pt``  → version « v18_322 », tag « bars »
    * ``hwm_v14_nojepa.pt``    → version « v14 », tag « nojepa »
    * ``hwm_lectaurep.pt``     → version None, tag « lectaurep »

    Args:
        filename: Nom du fichier (ex. ``hwm_v18_322_bars.pt``).

    Returns:
        Dictionnaire ``{"version": str | None, "tags": list[str]}``.
    """
    # Retirer l'extension .pt
    name = filename
    if name.endswith(".pt"):
        name = name[:-3]

    # Retirer le préfixe hwm_ (ou hwm)
    if name.startswith("hwm_"):
        name = name[4:]
    elif name.startswith("hwm"):
        name = name[3:]

    # Découper en segments séparés par des underscores
    parts = name.split("_") if name else []

    # La version est le premier segment de la forme v<chiffres>,
    # éventuellement suivi de segments purement numériques (ex. 322)
    version: str | None = None
    tag_start = 0
    for i, part in enumerate(parts):
        if re.match(r"^v\d+$", part):
            version = part
            tag_start = i + 1
            while tag_start < len(parts) and parts[tag_start].isdigit():
                version = f"{version}_{parts[tag_start]}"
                tag_start += 1
            break

    # Tout ce qui suit la version constitue les tags
    if version:
        tags = [t for t in parts[tag_start:] if t]
    else:
        tags = [t for t in parts if t]

    return {"version": version, "tags": tags}


# ---------------------------------------------------------------------------
# Lecture d'un checkpoint
# ---------------------------------------------------------------------------

def read_checkpoint(path: str) -> dict:
    """Lit les métadonnées d'un checkpoint SANS charger les poids du modèle.

    Le chargement s'effectue sur CPU (``map_location='cpu'``) avec
    ``weights_only=False`` car les checkpoints contiennent des objets
    personnalisés. Les champs ``model_state_dict`` et ``optimizer_state_dict``
    sont délibérément exclus du résultat : ils sont volumineux et feraient
    exploser la sérialisation JSON côté NiceGUI.

    Args:
        path: Chemin vers le fichier ``.pt``.

    Returns:
        Dictionnaire de métadonnées. En cas d'échec du chargement, les
        champs internes (epoch, loss, mode, …) restent à ``None`` et une
        clé ``"error"`` contenant le message d'exception est ajoutée.
    """
    p = Path(path)
    filename = p.name
    file_stat = p.stat()
    size_mb = file_stat.st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()

    parsed = parse_filename(filename)

    # Initialisation : les champs internes restent à None tant que
    # le chargement de torch n'a pas réussi
    info: dict = {
        "path": str(path),
        "filename": filename,
        "size_mb": round(size_mb, 1),
        "mtime": mtime,
        "version": parsed["version"],
        "epoch": None,
        "loss": None,
        "mode": None,
        "num_classes": None,
        "alphabet_size": None,
        "img_height": None,
        "embedding_dim": None,
        "tags": parsed["tags"],
    }

    try:
        data = torch.load(path, map_location="cpu", weights_only=False)

        # --- Champs directs ---
        info["epoch"] = data.get("epoch")
        info["loss"] = data.get("loss")
        info["mode"] = data.get("mode")

        # --- Config imbriquée ---
        config = data.get("config") or {}
        info["num_classes"] = config.get("num_classes")
        info["img_height"] = config.get("img_height")
        info["embedding_dim"] = config.get("embedding_dim")

        # --- Alphabet ---
        char_to_idx = data.get("char_to_idx") or {}
        info["alphabet_size"] = len(char_to_idx) if char_to_idx else None

        # IMPORTANT : on ne retourne JAMAIS model_state_dict ni
        # optimizer_state_dict — ils sont trop volumineux pour la
        # sérialisation JSON de NiceGUI

    except Exception as exc:
        info["error"] = str(exc)

    return info


# ---------------------------------------------------------------------------
# Scan d'un répertoire
# ---------------------------------------------------------------------------

def scan_checkpoints(root: str) -> list[dict]:
    """Scanne un répertoire et renvoie les métadonnées de tous les checkpoints.

    Recherche tous les fichiers ``hwm_*.pt`` dans ``root``, lit leurs
    métadonnées via :func:`read_checkpoint`, puis trie le résultat par date
    de modification décroissante (le plus récent en premier).

    Args:
        root: Chemin du répertoire contenant les checkpoints.

    Returns:
        Liste de dictionnaires de métadonnées, triée par ``mtime``.
    """
    pattern = os.path.join(root, "hwm_*.pt")
    files = glob.glob(pattern)
    checkpoints = [read_checkpoint(f) for f in files]
    checkpoints.sort(key=lambda c: c["mtime"], reverse=True)
    return checkpoints


# ---------------------------------------------------------------------------
# Utilitaires de formatage
# ---------------------------------------------------------------------------

def format_size(size_mb: float) -> str:
    """Formate une taille en mégaoctets vers une chaîne lisible.

    Args:
        size_mb: Taille en mégaoctets.

    Returns:
        Chaîne formatée, ex. ``« 850 Ko »``, ``« 27.5 Mo »``, ``« 1.2 Go »``.
    """
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} Go"
    if size_mb >= 1:
        return f"{size_mb:.1f} Mo"
    return f"{size_mb * 1024:.0f} Ko"


def format_tags(tags: list[str]) -> str:
    """Formate une liste de tags en chaîne d'affichage.

    Args:
        tags: Liste de tags (ex. ``["full", "bars"]``).

    Returns:
        Chaîne avec séparateur « · », ex. ``« full · bars »``.
    """
    return " · ".join(tags)
