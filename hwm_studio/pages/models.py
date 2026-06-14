"""Page de gestion des checkpoints — HWM Studio.

Affiche tous les fichiers ``hwm_*.pt`` à la racine du projet dans un
tableau triable avec leurs métadonnées (version, mode, epoch, loss,
alphabet, etc.) et des actions d'évaluation, fine-tuning et suppression.
"""

from __future__ import annotations

import os
from pathlib import Path

from nicegui import ui

from hwm_studio.checkpoint_info import scan_checkpoints, format_size, format_tags

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_checkpoints: list[dict] = []
_selected: dict | None = None


def build():
    """Construit la page Modèles."""
    global _checkpoints
    ui.label("Modèles (checkpoints)").classes("text-h4 q-mb-md")

    with ui.row().classes("items-center q-mb-md"):
        _checkpoints = _do_scan()
        total_size = sum(c.get("size_mb", 0) or 0 for c in _checkpoints)
        ui.label(
            f"{len(_checkpoints)} checkpoint(s) · {format_size(total_size)} au total"
        ).classes("text-grey")
        ui.button(icon="refresh", on_click=lambda: _refresh()).props(
            "flat round dense"
        ).tooltip("Rafraîchir")

    _table_section()
    _detail_section()


def _do_scan() -> list[dict]:
    try:
        return scan_checkpoints(str(PROJECT_ROOT))
    except Exception as exc:
        ui.notify(f"Erreur lors du scan: {exc}", type="negative")
        return []


def _table_section():
    colonnes = [
        {
            "name": "filename",
            "label": "Fichier",
            "field": "filename",
            "align": "left",
            "sortable": True,
        },
        {
            "name": "version",
            "label": "Version",
            "field": "version",
            "align": "center",
            "sortable": True,
        },
        {
            "name": "mode",
            "label": "Mode",
            "field": "mode",
            "align": "center",
            "sortable": True,
        },
        {
            "name": "epoch",
            "label": "Epoch",
            "field": "epoch",
            "align": "center",
            "sortable": True,
        },
        {
            "name": "loss",
            "label": "Loss",
            "field": "loss",
            "align": "center",
            "sortable": True,
        },
        {
            "name": "size",
            "label": "Taille",
            "field": "size",
            "align": "right",
            "sortable": True,
        },
        {
            "name": "tags",
            "label": "Tags",
            "field": "tags",
            "align": "left",
            "sortable": False,
        },
    ]

    lignes = []
    for ckpt in _checkpoints:
        err = ckpt.get("error")
        lignes.append(
            {
                "id": ckpt["path"],
                "filename": ckpt["filename"],
                "version": ckpt.get("version") or "—",
                "mode": ckpt.get("mode") or ("⚠️" if err else "—"),
                "epoch": str(ckpt["epoch"]) if ckpt.get("epoch") is not None else "—",
                "loss": (
                    f"{ckpt['loss']:.4f}"
                    if ckpt.get("loss") is not None
                    else "—"
                ),
                "size": format_size(ckpt.get("size_mb", 0) or 0),
                "tags": format_tags(ckpt.get("tags", [])),
                "_raw": ckpt,
            }
        )

    table = ui.table(
        columns=colonnes,
        rows=lignes,
        row_key="id",
    )
    table.props("flat dense rows-per-page-options=[10,20,50,100]")
    table.classes("w-full")
    table.on(
        "row-click",
        lambda e: _on_row_click(e.args[1]["id"] if len(e.args) > 1 else None),
    )


def _detail_section():
    global _selected
    _selected = None
    _detail_panel.refresh()


@ui.refreshable
def _detail_panel():
    if _selected is None:
        return
    ckpt = _selected

    with ui.card().classes("w-full q-mt-md"):
        with ui.row().classes("items-center justify-between"):
            ui.label(ckpt["filename"]).classes("text-h6")
            ui.button(icon="close", on_click=lambda: _close_detail()).props(
                "flat round dense"
            )

        err = ckpt.get("error")
        if err:
            ui.label(f"⚠️ Erreur de chargement: {err}").classes("text-negative")
            return

        with ui.row().classes("q-mt-sm"):
            ui.badge(f"{format_size(ckpt.get('size_mb', 0) or 0)}", color="primary")
            tags = ckpt.get("tags", [])
            if tags:
                ui.badge(format_tags(tags), color="info")

        with ui.row().classes("q-mt-md q-gutter-md"):
            _meta_item("Version", ckpt.get("version"))
            _meta_item("Mode", ckpt.get("mode"))
            _meta_item("Epoch", ckpt.get("epoch"))
            _meta_item("Loss", f"{ckpt['loss']:.4f}" if ckpt.get("loss") is not None else None)
            _meta_item("Classes", ckpt.get("num_classes"))
            _meta_item("Alphabet", ckpt.get("alphabet_size"))
            _meta_item("Hauteur img", ckpt.get("img_height"))
            _meta_item("Embed dim", ckpt.get("embedding_dim"))

        mtime = ckpt.get("mtime", "")
        if mtime:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(mtime)
                ui.label(
                    f"Modifié le {dt.strftime('%d/%m/%Y à %H:%M')}"
                ).classes("text-caption text-grey q-mt-sm")
            except Exception:
                pass

        ui.separator().classes("q-my-md")

        with ui.row().classes("q-gutter-sm"):
            ui.button(
                "Évaluer",
                icon="fact_check",
                on_click=lambda: ui.navigate.to("/?page=evaluate"),
            ).props("color=primary")
            ui.button(
                "Visualiser",
                icon="image",
                on_click=lambda: ui.navigate.to("/?page=visualize"),
            ).props("color=secondary flat")
            ui.button(
                "Fine-tuner",
                icon="model_training",
                on_click=lambda: ui.navigate.to("/?page=train"),
            ).props("color=positive flat")
            ui.button(
                "Supprimer",
                icon="delete",
                on_click=lambda: _confirm_delete(ckpt),
            ).props("color=negative flat")


def _meta_item(label: str, value):
    val_str = str(value) if value is not None else "—"
    with ui.column().classes("items-center"):
        ui.label(val_str).classes("text-subtitle1 text-weight-bold")
        ui.label(label).classes("text-caption text-grey")


def _on_row_click(row_id: str | None):
    global _selected
    if not row_id:
        return
    for c in _checkpoints:
        if c["path"] == row_id:
            _selected = c
            break
    _detail_panel.refresh()


def _close_detail():
    global _selected
    _selected = None
    _detail_panel.refresh()


def _confirm_delete(ckpt: dict):
    with ui.dialog() as dialog, ui.card():
        ui.label("Confirmer la suppression").classes("text-h6")
        ui.label(
            f"Voulez-vous vraiment supprimer « {ckpt['filename']} » "
            f"({format_size(ckpt.get('size_mb', 0) or 0)}) ?"
        ).classes("q-mt-sm")
        ui.label("Cette action est irréversible.").classes("text-negative text-caption")
        with ui.row().classes("q-mt-md justify-end"):
            ui.button("Annuler", on_click=dialog.close).props("flat")
            ui.button(
                "Supprimer",
                icon="delete",
                on_click=lambda: _do_delete(ckpt, dialog),
            ).props("color=negative")

    dialog.open()


def _do_delete(ckpt: dict, dialog):
    try:
        os.remove(ckpt["path"])
        ui.notify(f"Supprimé: {ckpt['filename']}", type="positive")
    except OSError as exc:
        ui.notify(f"Erreur: {exc}", type="negative")
    dialog.close()
    _close_detail()
    _refresh()


def _refresh():
    global _checkpoints
    _checkpoints = _do_scan()
    ui.navigate.to("/?page=models")
