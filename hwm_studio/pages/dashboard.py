"""Page Dashboard de HWM Studio.

Vue d'ensemble : état du GPU, nombre de checkpoints, volume des données
et processus actif. Toutes les opérations de scan sont enveloppées dans
des ``try/except`` afin que la page ne plante jamais.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from nicegui import ui

from hwm_studio import config_manager, runner, state
from hwm_studio.checkpoint_info import format_size, scan_checkpoints
from hwm_studio.data_scanner import get_corpus_summary, scan_all

# Racine du projet : parent du dossier parent de ``hwm_studio/``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ── GPU ──────────────────────────────────────────────────────────────────

def get_gpu_info() -> dict | None:
    """Interroge ``nvidia-smi`` et renvoie un résumé du GPU.

    Returns:
        Dictionnaire ``{"name", "mem_used_mb", "mem_total_mb",
        "util_pct"}``, ou ``None`` si ``nvidia-smi`` est absent ou
        échoue.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        name, mem_used, mem_total, util = (
            part.strip()
            for part in result.stdout.strip().splitlines()[0].split(",")
        )
        return {
            "name": name,
            "mem_used_mb": int(mem_used),
            "mem_total_mb": int(mem_total),
            "util_pct": int(util),
        }
    except Exception:
        return None


# ── Utilitaires de formatage ─────────────────────────────────────────────

def _fmt_mem_mb(mb: int) -> str:
    """Formate une quantité de mémoire (Mo) en « Mo » ou « Go »."""
    if mb >= 1024:
        return f"{mb / 1024:.1f} Go"
    return f"{mb} Mo"


def _fmt_lines(n: int) -> str:
    """Formate un nombre de lignes (« ~12k lignes » au-delà de 1000)."""
    if n >= 1000:
        return f"~{round(n / 1000)}k lignes"
    return f"{n} lignes"


# ── Cartes de statistiques ───────────────────────────────────────────────

def _card_gpu(gpu: dict | None) -> None:
    """Carte « GPU » : nom du matériel, VRAM utilisée/totale, utilisation."""
    with ui.card().classes("col column gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("memory", color="primary")
            ui.label("GPU").classes("text-h6 text-grey")
        if gpu is None:
            ui.label("N/A").classes("text-h2")
            ui.label("GPU indisponible").classes("text-grey")
            return
        ui.label(gpu["name"]).classes("text-h6")
        ui.label(
            f"{_fmt_mem_mb(gpu['mem_used_mb'])} / {_fmt_mem_mb(gpu['mem_total_mb'])}"
        ).classes("text-h5 text-primary")
        ui.label(f"{gpu['util_pct']} % d'utilisation").classes("text-grey")


def _card_models(checkpoints: list[dict] | None, error: str | None) -> None:
    """Carte « Modèles » : nombre de checkpoints et taille totale."""
    with ui.card().classes("col column gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("storage", color="primary")
            ui.label("Modèles").classes("text-h6 text-grey")
        if error is not None:
            ui.label("N/A").classes("text-h2")
            ui.label(error).classes("text-grey")
            return
        count = len(checkpoints) if checkpoints else 0
        total_mb = sum(c.get("size_mb", 0) for c in (checkpoints or []))
        ui.label(str(count)).classes("text-h2")
        ui.label(format_size(total_mb) + " au total").classes("text-grey")


def _card_data(total_dirs: int, total_lines: int, nb_sel: int) -> None:
    """Carte « Données » : nombre de dossiers, lignes estimées, sélection."""
    with ui.card().classes("col column gap-1"):
        with ui.row().classes("items-center gap-2"):
            ui.icon("folder_open", color="primary")
            ui.label("Données").classes("text-h6 text-grey")
        ui.label(str(total_dirs)).classes("text-h2")
        ui.label(f"dossiers · {_fmt_lines(total_lines)}").classes("text-grey")
        ui.label(f"{nb_sel} sélectionnée(s) pour l'entraînement").classes(
            "text-caption text-grey-6"
        )


# ── Sections de détail ───────────────────────────────────────────────────

def _render_sources(
    sources: list[dict], corpus_summary: dict[str, dict]
) -> None:
    """Affiche le résumé des sources de données, regroupées par corpus."""
    ui.label("Sources de données").classes("text-h5 q-mt-md")
    if not sources:
        with ui.column().classes("gap-0 q-mt-xs"):
            ui.label("Aucune source configurée.").classes("text-grey")
            ui.label(
                "Ajoutez des dossiers ALTO depuis la page « Sources »."
            ).classes("text-caption text-grey-6")
        return
    with ui.column().classes("gap-1 q-mt-sm"):
        for corpus, info in sorted(corpus_summary.items()):
            ui.label(
                f"• {corpus} — {info['dir_count']} dossier(s), "
                f"{_fmt_lines(info['line_count_est'])}"
            ).classes("text-body1")


def _render_process() -> None:
    """Affiche l'état du processus actif (carte mise en valeur si en cours)."""
    ui.label("Processus actif").classes("text-h5 q-mt-md")
    if runner.is_running():
        commande = runner.get_active_command() or "En cours…"
        with ui.card().classes("column gap-1 bg-primary text-white"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("play_circle")
                ui.label("En cours d'exécution").classes("text-subtitle2")
            ui.label(commande).classes("text-body1").style(
                "font-family: monospace; word-break: break-all"
            )
    else:
        ui.label("Aucun processus actif.").classes("text-grey")


# ── Contenu rafraîchissable ──────────────────────────────────────────────

@ui.refreshable
def _dashboard_content() -> None:
    """Construit l'ensemble du contenu du dashboard (rafraîchissable)."""

    # ── GPU ──
    gpu = get_gpu_info()

    # ── Checkpoints ──
    checkpoints: list[dict] | None = None
    ckpt_error: str | None = None
    try:
        checkpoints = scan_checkpoints(str(PROJECT_ROOT))
    except Exception:
        ckpt_error = "Lecture des checkpoints impossible"

    # ── Données ──
    sources: list[dict] = []
    corpus_summary: dict[str, dict] = {}
    total_lines = 0
    try:
        sources = config_manager.get_data_sources()
        if sources:
            # scan_all : totaux bruts ; get_corpus_summary : regroupement.
            scans = scan_all(sources)
            corpus_summary = get_corpus_summary(sources)
            total_lines = sum(
                s.get("line_count_est", 0) for s in scans.values()
            )
    except Exception:
        sources = []

    total_dirs = len(sources)
    try:
        nb_sel = len(state.get_selected_sources())
    except Exception:
        nb_sel = 0

    # ── Cartes côte à côte ──
    with ui.row().classes("w-full items-stretch"):
        _card_gpu(gpu)
        _card_models(checkpoints, ckpt_error)
        _card_data(total_dirs, total_lines, nb_sel)

    # ── Détails ──
    _render_sources(sources, corpus_summary)
    _render_process()


# ── Construction de la page ──────────────────────────────────────────────

def build() -> None:
    """Construit la page Dashboard."""
    with ui.column().classes("w-full q-pa-lg gap-2"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("Dashboard").classes("text-h4")
            ui.button(
                "Rafraîchir",
                icon="refresh",
                on_click=_dashboard_content.refresh,
            ).props("outline")
        _dashboard_content()
