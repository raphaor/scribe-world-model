"""Gestionnaire de configuration JSON pour HWM Studio.

Lit et écrit le fichier ``hwm_studio_config.json`` à la racine du projet.
Ce module gère la configuration propre au GUI (catalogue de sources de
données, préférences) — il ne modifie **pas** le ``config.py`` existant.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "hwm_studio_config.json"

DEFAULT_CONFIG = {
    "data_sources": {},          # {dir_path: {"label": str, "corpus": str, "custom": bool}}
    "alto_scan_root": "D:/OCR_genealogie/Alto",
    "default_gt_mode": "original",   # "original" | "gt"
    "recent_training": {},       # Derniers paramètres d'entraînement utilisés
}


# ── Utilitaires internes ──────────────────────────────────────────────────

def _norm(path: str) -> str:
    """Normalise un chemin en remplaçant les antislashs par des ``/``."""
    return str(path).replace("\\", "/")


def _derive_label(path: str) -> str:
    """Déduit un libellé lisible à partir du nom du dossier.

    Les underscores sont remplacés par des espaces et le résultat est
    mis en titre (title case).
    """
    name = Path(path).name
    return name.replace("_", " ").title()


def _fresh_defaults() -> dict:
    """Renvoie une copie profonde de ``DEFAULT_CONFIG``."""
    return json.loads(json.dumps(DEFAULT_CONFIG))


def _merge_dicts(base: dict, override: dict) -> dict:
    """Fusionne récursivement ``override`` dans ``base`` (modification en place).

    Les clés manquantes de ``base`` sont comblées par ``override``, et
    inversement ; les sous-dictionnaires sont fusionnés en profondeur.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _import_alto_dirs() -> list[str]:
    """Importe ``ALTO_DIRS`` depuis ``config.py`` à la racine du projet.

    L'import est volontairement défensif (``try/except``) : si le fichier
    ne peut pas être chargé — chemin absent, erreur d'exécution, etc. —
    on renvoie une liste vide plutôt que de planter le GUI.
    """
    config_file = PROJECT_ROOT / "config.py"
    if not config_file.is_file():
        return []
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_hwm_config_ref", config_file)
        if spec is None or spec.loader is None:
            return []
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return list(getattr(mod, "ALTO_DIRS", []))
    except Exception:
        return []


# ── API publique ──────────────────────────────────────────────────────────

def load_config() -> dict:
    """Charge la configuration depuis le fichier JSON.

    * Si le fichier n'existe pas encore (premier lancement), il est créé
      avec les valeurs par défaut et pré-rempli à partir de ``ALTO_DIRS``
      du ``config.py`` du projet.
    * Si le fichier existe, ses valeurs sont fusionnées avec les
      valeurs par défaut afin que les clés manquantes soient comblées.
    """
    merged = _fresh_defaults()

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                _merge_dicts(merged, loaded)
        except (json.JSONDecodeError, OSError):
            # Fichier illisible : on conserve les valeurs par défaut.
            pass
    else:
        # Premier lancement : pré-remplir le catalogue avec ALTO_DIRS.
        for raw_dir in _import_alto_dirs():
            dir_path = _norm(raw_dir)
            merged["data_sources"][dir_path] = {
                "label": _derive_label(dir_path),
                "corpus": "Autre",
                "custom": False,
            }
        save_config(merged)

    return merged


def save_config(config: dict) -> None:
    """Écrit la configuration au format JSON (``indent=2``, UTF-8 pur)."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_data_sources() -> list[dict]:
    """Retourne la liste des sources de données enregistrées.

    Chaque entrée est un dict :
    ``{"path": str, "label": str, "corpus": str, "custom": bool}``
    """
    config = load_config()
    sources = []
    for dir_path, info in config.get("data_sources", {}).items():
        if not isinstance(info, dict):
            info = {}
        sources.append({
            "path": dir_path,
            "label": info.get("label", ""),
            "corpus": info.get("corpus", "Autre"),
            "custom": info.get("custom", True),
        })
    return sources


def add_data_source(
    path: str,
    label: str = "",
    corpus: str = "Autre",
    custom: bool = True,
) -> None:
    """Ajoute ou met à jour une source de données dans la configuration.

    Si ``label`` est vide, un libellé est déduit automatiquement du nom
    du dossier (underscores remplacés par des espaces, title case).
    """
    dir_path = _norm(path)
    if not label:
        label = _derive_label(dir_path)

    config = load_config()
    config.setdefault("data_sources", {})[dir_path] = {
        "label": label,
        "corpus": corpus,
        "custom": custom,
    }
    save_config(config)


def remove_data_source(path: str) -> None:
    """Retire une source de données de la configuration."""
    dir_path = _norm(path)
    config = load_config()
    config.get("data_sources", {}).pop(dir_path, None)
    save_config(config)


def auto_scan_sources() -> list[str]:
    """Scanne ``alto_scan_root`` à la recherche de sous-dossiers ALTO.

    Ne renvoie que les sous-dossiers **immédiats** contenant au moins un
    fichier ``.xml`` (en excluant ``METS.xml``, comparaison insensible à
    la casse). Les chemins découverts ne sont **pas** ajoutés à la
    configuration — l'appelant décide de les intégrer ou non.
    """
    config = load_config()
    root = config.get("alto_scan_root", "")
    if not root or not os.path.isdir(root):
        return []

    found: list[str] = []
    for entry in sorted(os.listdir(root)):
        subdir = os.path.join(root, entry)
        if not os.path.isdir(subdir):
            continue
        xml_files = glob.glob(os.path.join(subdir, "*.xml"))
        if any(os.path.basename(f).upper() != "METS.XML" for f in xml_files):
            found.append(_norm(subdir))
    return found


def get_default_gt_mode() -> str:
    """Retourne le mode GT par défaut (``"original"`` ou ``"gt"``)."""
    return load_config().get("default_gt_mode", "original")


def save_recent_training(params: dict) -> None:
    """Mémorise les derniers paramètres d'entraînement utilisés."""
    config = load_config()
    config["recent_training"] = params
    save_config(config)
