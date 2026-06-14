"""État partagé de l'application HWM Studio."""

from __future__ import annotations

# Sources de données sélectionnées pour le training/eval/etc.
# {source_path: "original" | "gt"}
selected_sources: dict[str, str] = {}

# ID du processus actif (ou None)
active_process_id: str | None = None

# Dernière commande lancée (pour affichage)
last_command: str = ""

# Callback pour rafraîchir l'affichage quand selected_sources change
_on_sources_changed: callable = None


# --------------------------------------------------------------------------- #
#  Sources sélectionnées
# --------------------------------------------------------------------------- #

def set_selected_sources(sources: dict[str, str]) -> None:
    """Met à jour les sources sélectionnées et notifie les observateurs.

    Args:
        sources: dictionnaire ``{source_path: "original" | "gt"}``.
    """
    global selected_sources
    selected_sources = dict(sources)
    if _on_sources_changed is not None:
        _on_sources_changed()


def get_selected_sources() -> dict[str, str]:
    """Retourne une copie de la sélection courante de sources."""
    return dict(selected_sources)


def register_sources_callback(cb: callable) -> None:
    """Enregistre un callback appelé quand ``selected_sources`` change.

    Args:
        cb: callable sans argument invoqué après chaque mise à jour.
    """
    global _on_sources_changed
    _on_sources_changed = cb


# --------------------------------------------------------------------------- #
#  Résolution des dossiers ALTO
# --------------------------------------------------------------------------- #

def get_resolved_alto_dirs() -> list[str]:
    """Résout les dossiers ALTO réels à passer aux scripts.

    Importe ``gt_staging`` à l'intérieur de la fonction pour éviter les
    imports circulaires. Si ``gt_staging`` n'est pas encore disponible,
    renvoie simplement les chemins bruts des sources sélectionnées.

    Returns:
        Liste des dossiers ALTO résolus.
    """
    try:
        from hwm_studio import gt_staging
    except ImportError:
        # Module non encore implémenté : on retombe sur les chemins bruts.
        return list(selected_sources.keys())
    return gt_staging.resolve_alto_dirs(selected_sources)
