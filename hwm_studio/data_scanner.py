"""
Analyse rapide des répertoires de données ALTO pour HWM Studio.

Parcourt les répertoires contenant les fichiers ALTO (*.xml + *.jpg),
estime le volume de lignes de texte, mesure la couverture de ground truth
et lit la progression d'annotation stockée dans *_gt.progress.json.
"""

import os
import glob
import json
from pathlib import Path


# ─── Comptage de lignes ─────────────────────────────────────────────────

def count_textlines_quick(xml_path: str) -> int:
    """Compte les occurrences de ``<TextLine`` dans un fichier XML.

    Lecture brute du fichier texte — aucun parseur XML, rapide et
    approximatif. Suffisant pour une estimation de volume.

    Args:
        xml_path: chemin vers un fichier ALTO .xml.

    Returns:
        Nombre d'occurrences de la chaîne ``<TextLine``.
    """
    try:
        content = Path(xml_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return content.count("<TextLine")


def count_textlines_sample(dir_path: str, sample_size: int = 5) -> int:
    """Estime le nombre total de lignes en échantillonnant des fichiers XML.

    Lit les ``sample_size`` premiers fichiers .xml du répertoire (hors
    METS.xml et *_gt.xml), compte les ``<TextLine`` de chacun, puis
    extrapole à l'ensemble des fichiers.

    Args:
        dir_path: répertoire contenant les fichiers ALTO.
        sample_size: nombre de fichiers à échantillonner.

    Returns:
        Estimation du nombre total de lignes, ou 0 si aucun fichier.
    """
    all_xml = sorted(
        f for f in glob.glob(os.path.join(dir_path, "*.xml"))
        if os.path.basename(f) != "METS.xml"
        and not os.path.splitext(f)[0].endswith("_gt")
    )
    total_xml_count = len(all_xml)
    if total_xml_count == 0:
        return 0

    sample = all_xml[:sample_size]
    total_sample_lines = sum(count_textlines_quick(f) for f in sample)
    sampled_count = len(sample)

    if sampled_count == 0:
        return 0
    return int(total_sample_lines / sampled_count * total_xml_count)


# ─── Scan d'un répertoire ───────────────────────────────────────────────

def scan_directory(path: str) -> dict:
    """Analyse un répertoire de données ALTO.

    Args:
        path: chemin du répertoire à analyser.

    Returns:
        Dictionnaire avec les statistiques du répertoire::

            {
                "path": str,
                "exists": bool,
                "xml_count": int,       # *.xml hors METS.xml et *_gt.xml
                "jpg_count": int,       # *.jpg
                "gt_count": int,        # *_gt.xml
                "line_count_est": int,  # estimation des lignes de texte
                "gt_coverage": float,   # gt_count / max(xml_count, 1)
                "progress": dict | None,
            }
    """
    result = {
        "path": path,
        "exists": False,
        "xml_count": 0,
        "jpg_count": 0,
        "gt_count": 0,
        "line_count_est": 0,
        "gt_coverage": 0.0,
        "progress": None,
    }

    if not os.path.isdir(path):
        return result

    result["exists"] = True

    # Fichiers XML sources : *.xml hors METS.xml et *_gt.xml
    all_xml = [
        f for f in glob.glob(os.path.join(path, "*.xml"))
        if os.path.basename(f) != "METS.xml"
        and not os.path.splitext(f)[0].endswith("_gt")
    ]
    result["xml_count"] = len(all_xml)

    # Images JPG
    result["jpg_count"] = len(glob.glob(os.path.join(path, "*.jpg")))

    # Fichiers de ground truth (*_gt.xml)
    result["gt_count"] = len(glob.glob(os.path.join(path, "*_gt.xml")))

    # Estimation du nombre de lignes (échantillonnage + extrapolation)
    result["line_count_est"] = count_textlines_sample(path)

    # Couverture de ground truth (0.0 à 1.0)
    result["gt_coverage"] = result["gt_count"] / max(result["xml_count"], 1)

    # Progression d'annotation depuis le premier *_gt.progress.json trouvé
    progress_files = sorted(glob.glob(os.path.join(path, "*_gt.progress.json")))
    if progress_files:
        try:
            with open(progress_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
            result["progress"] = {
                "accepted": len(data.get("accepted", {})),
                "last_page": data.get("last_page", 0),
            }
        except (OSError, json.JSONDecodeError):
            result["progress"] = None

    return result


# ─── Scan multi-répertoires ─────────────────────────────────────────────

def scan_all(sources: list[dict]) -> dict[str, dict]:
    """Analyse plusieurs répertoires de données.

    Args:
        sources: liste de dictionnaires ``{"path": str, "label": str,
            "corpus": str}``.

    Returns:
        Dictionnaire ``{path: scan_directory(path)}``. Un répertoire en
        erreur ne provoque pas d'exception : un dictionnaire minimal est
        retourné avec ``exists=False``.
    """
    results = {}
    for source in sources:
        path = source.get("path", "")
        try:
            results[path] = scan_directory(path)
        except Exception:
            results[path] = {
                "path": path,
                "exists": False,
                "xml_count": 0,
                "jpg_count": 0,
                "gt_count": 0,
                "line_count_est": 0,
                "gt_coverage": 0.0,
                "progress": None,
            }
    return results


# ─── Résumé par corpus ──────────────────────────────────────────────────

def get_corpus_summary(sources: list[dict]) -> dict[str, dict]:
    """Regroupe les sources par corpus et agrège les statistiques.

    Args:
        sources: liste de dictionnaires ``{"path": str, "label": str,
            "corpus": str}``.

    Returns:
        Dictionnaire ``{corpus: {"dir_count": int, "line_count_est": int,
        "xml_count": int}}``.
    """
    scans = scan_all(sources)

    summary: dict[str, dict] = {}
    for source in sources:
        corpus = source.get("corpus", "Inconnu")
        path = source.get("path", "")
        scan = scans.get(path, {})

        if corpus not in summary:
            summary[corpus] = {
                "dir_count": 0,
                "line_count_est": 0,
                "xml_count": 0,
            }

        summary[corpus]["dir_count"] += 1
        summary[corpus]["line_count_est"] += scan.get("line_count_est", 0)
        summary[corpus]["xml_count"] += scan.get("xml_count", 0)

    return summary
