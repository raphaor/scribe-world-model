"""Mise en scène (« staging ») des répertoires ALTO pour le mode GT augmentée.

Lorsque l'utilisateur sélectionne « GT augmentée » pour une source de
données, ce module crée un répertoire de *staging* dans lequel les
fichiers ``page_gt.xml`` remplacent les ``page.xml`` originaux.

C'est nécessaire car les scripts d'entraînement associent le XML au JPG
par nom de base : ``_parse_page`` fait ``jpg_path = xml_path.replace(
".xml", ".jpg")``. Si le XML est ``page_gt.xml``, il cherche donc
``page_gt.jpg`` qui n'existe pas — les corrections GT seraient invisibles
pour l'entraînement.

Le répertoire de staging résout cela en produisant, pour chaque page :
  * ``page.xml``  → copie de ``page_gt.xml`` (s'il existe) ou de l'original ;
  * ``page.jpg``  → lien symbolique vers le ``page.jpg`` source (les JPG
    sont volumineux, les liens symboliques évitent la duplication).
"""

from __future__ import annotations

import glob
import hashlib
import json  # noqa: F401  (réservé pour extension future)
import os
import shutil
from pathlib import Path

# ── Constantes ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGING_BASE = PROJECT_ROOT / ".cache_alto"


# ── Utilitaires internes ──────────────────────────────────────────────────

def _norm(path) -> str:
    """Normalise un chemin en remplaçant les antislashs par des ``/``."""
    return str(path).replace("\\", "/")


def _compute_hash(source_dir: str) -> str:
    """Calcule une empreinte MD5 identifiant l'état GT courant d'un dossier.

    Les entrées du hash sont : le chemin du dossier source, puis la liste
    triée de tous les fichiers ``*_gt.xml`` avec leur date de modification
    (mtime). Ainsi, dès qu'un fichier GT est ajouté ou modifié, l'empreinte
    change et un nouveau répertoire de staging est créé.

    Args:
        source_dir: chemin du répertoire source ALTO.

    Returns:
        Les 12 premiers caractères du hexdigest.
    """
    h = hashlib.md5()
    h.update(source_dir.encode("utf-8"))

    gt_files = sorted(glob.glob(os.path.join(source_dir, "*_gt.xml")))
    for gt in gt_files:
        h.update(os.path.basename(gt).encode("utf-8"))
        try:
            mtime = os.path.getmtime(gt)
        except OSError:
            mtime = 0
        h.update(str(mtime).encode("utf-8"))

    return h.hexdigest()[:12]


def get_staging_path(source_dir: str) -> str:
    """Retourne le chemin du répertoire de staging d'un dossier source.

    Le répertoire n'est **pas** créé par cette fonction — elle sert
    uniquement à déterminer son emplacement.

    Format : ``STAGING_BASE / "staging_gt_{hash}"``.
    """
    return _norm(STAGING_BASE / f"staging_gt_{_compute_hash(source_dir)}")


# ── Création du staging ───────────────────────────────────────────────────

def create_staging(source_dir: str) -> str:
    """Crée le répertoire de staging pour un dossier source.

    Pour chaque fichier ``*.xml`` du dossier source (hors ``METS.xml`` et
    ``*_gt.xml``) :
      * si ``{base}_gt.xml`` existe, il est copié dans le staging sous le
        nom ``{base}.xml`` (la version GT remplace l'original) ;
      * sinon, l'original ``{base}.xml`` est copié tel quel ;
      * le ``{base}.jpg`` correspondant est lié par lien symbolique vers
        le JPG source, avec repli sur une copie si le lien symbolique
        échoue (Windows sans Mode développeur).

    Si le répertoire de staging existe déjà (cache valide), il est
    retourné tel quel sans recréation.

    Args:
        source_dir: chemin du répertoire source ALTO.

    Returns:
        Le chemin du répertoire de staging (en slashes ``/``). En cas
        d'échec (dossier source absent, erreur de création), le chemin
        source d'origine est retourné en repli.
    """
    source_dir = _norm(source_dir)

    # Dossier source absent : rien à mettre en scène, on retourne tel quel.
    if not os.path.isdir(source_dir):
        print(f"[gt_staging] Dossier source introuvable : {source_dir}")
        return source_dir

    # S'assurer que la base de staging existe.
    try:
        STAGING_BASE.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[gt_staging] Impossible de créer {STAGING_BASE} : {exc}")
        return source_dir

    staging_dir = STAGING_BASE / f"staging_gt_{_compute_hash(source_dir)}"

    # Cache : le staging existe déjà, on le retourne directement.
    if staging_dir.is_dir():
        return _norm(staging_dir)

    # Création du répertoire de staging.
    try:
        staging_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        print(f"[gt_staging] Échec de création du staging {staging_dir} : {exc}")
        return source_dir

    # Sélection des XML sources : *.xml hors METS.xml et *_gt.xml.
    source_xml_files = [
        f for f in glob.glob(os.path.join(source_dir, "*.xml"))
        if os.path.basename(f).upper() != "METS.XML"
        and not os.path.splitext(f)[0].endswith("_gt")
    ]

    xml_done = 0
    jpg_done = 0
    jpg_copied = 0  # nombre de JPG réellement copiés (repli)

    for xml_path in sorted(source_xml_files):
        basename = os.path.basename(xml_path)          # ex. "page1.xml"
        stem = os.path.splitext(basename)[0]           # ex. "page1"
        gt_path = os.path.join(source_dir, f"{stem}_gt.xml")

        # XML de destination dans le staging (toujours {stem}.xml).
        dst_xml = staging_dir / basename

        try:
            if os.path.isfile(gt_path):
                shutil.copy2(gt_path, dst_xml)
            else:
                shutil.copy2(xml_path, dst_xml)
            xml_done += 1
        except OSError as exc:
            print(f"[gt_staging] Copie XML échouée ({basename}) : {exc}")

        # JPG correspondant : lien symbolique vers la source.
        src_jpg = os.path.join(source_dir, f"{stem}.jpg")
        dst_jpg = staging_dir / f"{stem}.jpg"

        if not os.path.isfile(src_jpg):
            continue

        try:
            os.symlink(src_jpg, dst_jpg)
            jpg_done += 1
        except (OSError, NotImplementedError) as exc:
            # Windows sans Mode développeur / privilèges admin :
            # repli sur une copie matérielle du JPG.
            print(
                f"[gt_staging] Lien symbolique impossible pour "
                f"{stem}.jpg ({exc.__class__.__name__}) — copie de repli."
            )
            try:
                shutil.copy2(src_jpg, dst_jpg)
                jpg_done += 1
                jpg_copied += 1
            except OSError as exc2:
                print(f"[gt_staging] Copie JPG échouée ({stem}.jpg) : {exc2}")

    if jpg_copied:
        print(
            f"Staging GT: copié {xml_done} XML, "
            f"{jpg_done} JPG (dont {jpg_copied} par copie de repli)"
        )
    else:
        print(f"Staging GT: copié {xml_done} XML, {jpg_done} JPG")

    return _norm(staging_dir)


# ── Résolution des répertoires ALTO ───────────────────────────────────────

def resolve_alto_dirs(selection: dict[str, str]) -> list[str]:
    """Résout les répertoires ALTO finaux à passer aux scripts.

    Cette fonction clé est appelée par les pages avant de lancer un
    script (entraînement, reconnaissance, etc.).

    Args:
        selection: dictionnaire ``{source_path: "original" | "gt"}``
            indiquant, pour chaque source sélectionnée, le mode GT
            souhaité.

    Returns:
        Liste des chemins de répertoires résolus. Pour le mode
        ``"original"``, le chemin source est ajouté tel quel ; pour le
        mode ``"gt"``, le répertoire de staging créé est ajouté. Une
        sélection vide renvoie une liste vide.
    """
    if not selection:
        return []

    resolved: list[str] = []
    for source_path, mode in selection.items():
        if mode == "gt":
            resolved.append(create_staging(source_path))
        else:
            # "original" ou toute autre valeur : chemin source direct.
            resolved.append(_norm(source_path))
    return resolved


# ── Nettoyage ─────────────────────────────────────────────────────────────

def cleanup_staging() -> None:
    """Supprime tous les répertoires de staging (``staging_gt_*``).

    Parcourt ``STAGING_BASE`` à la recherche des dossiers dont le nom
    commence par ``staging_gt_`` et les supprime récursivement. Affiche
    le nombre de répertoires supprimés.
    """
    if not STAGING_BASE.is_dir():
        print("[gt_staging] Aucun répertoire de cache à nettoyer.")
        return

    removed = 0
    for entry in glob.glob(os.path.join(_norm(STAGING_BASE), "staging_gt_*")):
        if os.path.isdir(entry):
            try:
                shutil.rmtree(entry)
                removed += 1
            except OSError as exc:
                print(f"[gt_staging] Suppression échouée ({entry}) : {exc}")

    print(f"[gt_staging] {removed} répertoire(s) de staging supprimé(s).")


def cleanup_staging_for(source_dir: str) -> None:
    """Supprime le répertoire de staging associé à un dossier source.

    Args:
        source_dir: chemin du répertoire source ALTO.
    """
    staging_dir = get_staging_path(source_dir)
    if os.path.isdir(staging_dir):
        try:
            shutil.rmtree(staging_dir)
            print(f"[gt_staging] Staging supprimé : {staging_dir}")
        except OSError as exc:
            print(f"[gt_staging] Suppression échouée ({staging_dir}) : {exc}")
    else:
        print(f"[gt_staging] Aucun staging à supprimer pour : {source_dir}")
