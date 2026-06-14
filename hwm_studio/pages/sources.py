"""Page « Sources de données » de HWM Studio (NiceGUI).

Catalogue central des répertoires ALTO. L'utilisateur peut :
  * voir toutes les sources connues, regroupées par corpus ;
  * consulter les statistiques temps réel de chaque répertoire
    (nombre de XML, JPG, GT, estimation de lignes, couverture GT) ;
  * sélectionner/désélectionner des sources pour le training/eval ;
  * basculer chaque source entre les modes « Original » et « GT augmentée » ;
  * ajouter de nouvelles sources (détection auto + saisie manuelle) ;
  * supprimer des sources existantes.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from hwm_studio import config_manager, state
from hwm_studio.data_scanner import scan_directory, get_corpus_summary  # noqa: F401  (get_corpus_summary exposé pour cohérence de l'API)


# ── Cache des scans ──────────────────────────────────────────────────────
# {path: scan_directory(path)} — rafraîchi à la construction de la page et
# quand l'utilisateur clique sur « Scan ». Les ajouts remplissent le cache
# à la volée (miss -> scan immédiat).

_scan_cache: dict[str, dict] = {}


# ── Helpers de mise en forme ─────────────────────────────────────────────

def _fmt_lines(n: int) -> str:
    """Formate un nombre de lignes estimé de façon lisible.

    Au-dessus de 1000 lignes, on utilise une notation en milliers
    (ex. ``3200`` -> ``~3.2k``).
    """
    if n >= 1000:
        return f"~{n / 1000:.1f}k"
    return f"~{n}"


def _coverage_color(coverage: float) -> str:
    """Couleur de la barre de couverture GT selon le taux.

    * ``> 50 %`` : ``positive`` (vert) ;
    * ``> 0 %``  : ``warning`` (orange) ;
    * ``0 %``    : ``grey``.
    """
    if coverage > 0.5:
        return "positive"
    if coverage > 0.0:
        return "warning"
    return "grey"


def _empty_scan(path: str, error: str | None = None) -> dict:
    """Retourne un résultat de scan vide (répertoire absent ou en erreur)."""
    return {
        "path": path,
        "exists": False,
        "xml_count": 0,
        "jpg_count": 0,
        "gt_count": 0,
        "line_count_est": 0,
        "gt_coverage": 0.0,
        "progress": None,
        "error": error,
    }


def _refresh_cache(sources: list[dict]) -> None:
    """Re-scanne intégralement les sources et met à jour ``_scan_cache``."""
    _scan_cache.clear()
    for source in sources:
        path = source.get("path", "")
        if not path:
            continue
        try:
            _scan_cache[path] = scan_directory(path)
        except Exception as exc:  # noqa: BLE001 — on ne veut jamais planter le GUI
            _scan_cache[path] = _empty_scan(path, str(exc))


def _get_scan(path: str) -> dict:
    """Retourne le scan en cache pour ``path``, ou le calcule à la volée."""
    if path not in _scan_cache:
        try:
            _scan_cache[path] = scan_directory(path)
        except Exception as exc:  # noqa: BLE001
            _scan_cache[path] = _empty_scan(path, str(exc))
    return _scan_cache[path]


# ── Construction de la page ──────────────────────────────────────────────

def build():
    """Construit la page Sources de données."""
    # Premier scan à l'ouverture de la page.
    _refresh_cache(config_manager.get_data_sources())

    # ── En-tête : titre + actions ──
    with ui.row().classes("w-full items-center justify-between q-pa-md"):
        ui.label("Sources de données").classes("text-h5")
        with ui.row().classes("items-center gap-2"):
            ui.button("+ Ajouter", icon="add",
                      on_click=lambda: _open_add_dialog()).props("color=primary")
            ui.button("Scanner", icon="refresh",
                      on_click=lambda: _do_scan()).props("outline")

    # ── Rafraîchissables ──
    # `content` : les groupes par corpus (reconstruit après ajout/suppression/scan).
    # `summary` : la barre de résumé (reconstruit après chaque changement de sélection).

    @ui.refreshable
    def content():
        sources = config_manager.get_data_sources()
        sel = state.get_selected_sources()
        default_gt = config_manager.get_default_gt_mode()

        if not sources:
            ui.label("Aucune source enregistrée. Cliquez sur « + Ajouter » pour en ajouter."
                     ).classes("text-grey q-pa-md")
            return

        # Regroupement par corpus, ordre alphabétique.
        groups: dict[str, list[dict]] = {}
        for s in sources:
            corpus = s.get("corpus") or "Autre"
            groups.setdefault(corpus, []).append(s)

        with ui.column().classes("w-full q-px-md q-pb-md gap-2"):
            for corpus in sorted(groups):
                items = sorted(groups[corpus], key=lambda x: x.get("label", ""))
                sel_in = sum(1 for s in items if s["path"] in sel)
                title = f"{corpus}  ({sel_in}/{len(items)} sélectionné(s))"
                with ui.expansion(title).classes("w-full"):
                    for s in items:
                        _build_source_card(s, sel, default_gt)

    @ui.refreshable
    def summary():
        sel = state.get_selected_sources()
        n_dirs = len(sel)
        total_lines = 0
        gt_count = 0
        for path, mode in sel.items():
            scan = _get_scan(path)
            total_lines += scan.get("line_count_est", 0)
            if mode == "gt":
                gt_count += 1
        with ui.row().classes("w-full items-center q-pa-md bordered"):
            ui.icon("summarize", color="primary")
            ui.label(
                f"Résumé : {n_dirs} dossier(s) sélectionné(s) · "
                f"{_fmt_lines(total_lines)} lignes · {gt_count} en GT"
            ).classes("text-body1")

    # ── Actions locales ──

    def _do_scan():
        _refresh_cache(config_manager.get_data_sources())
        content.refresh()
        summary.refresh()
        ui.notify("Scan terminé.", type="positive")

    def _build_source_card(source: dict, sel: dict, default_gt: str):
        path = source["path"]
        label = source.get("label") or Path(path).name.replace("_", " ").title()
        scan = _get_scan(path)
        exists = scan.get("exists", False)
        gt_count = scan.get("gt_count", 0)
        xml_count = scan.get("xml_count", 0)
        jpg_count = scan.get("jpg_count", 0)
        line_est = scan.get("line_count_est", 0)
        coverage = scan.get("gt_coverage", 0.0)
        error = scan.get("error")
        mode = sel.get(path, default_gt)

        with ui.card().classes("w-full"):
            # Ligne 1 : checkbox + libellé ... toggle + suppression
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                with ui.row().classes("items-center gap-2"):
                    ui.checkbox(
                        "",
                        value=(path in sel),
                        on_change=_on_checkbox(path, default_gt),
                    )
                    ui.label(label).classes("text-subtitle1")
                    if not exists:
                        ui.label("Dossier introuvable"
                                 ).classes("text-negative text-caption")
                    elif error:
                        ui.label(f"Erreur de scan : {error}"
                                 ).classes("text-negative text-caption")
                with ui.row().classes("items-center gap-2"):
                    if exists and gt_count > 0:
                        ui.toggle(
                            {"original": "Original", "gt": "GT"},
                            value=mode,
                            on_change=_on_gt(path),
                        )
                    ui.button(
                        icon="delete",
                        on_click=_on_delete(path),
                    ).props("flat dense round color=negative")

            # Ligne 2 : statistiques
            if exists and not error:
                if xml_count == 0:
                    ui.label("Aucun fichier XML trouvé."
                             ).classes("text-caption text-grey")
                else:
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.label(
                            f"{xml_count} XML · {jpg_count} JPG · {gt_count} GT · "
                            f"{_fmt_lines(line_est)} lignes · GT : {int(coverage * 100)}%"
                        ).classes("text-caption text-grey")

                    # Ligne 3 : barre de couverture GT
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.linear_progress(
                            value=coverage,
                            color=_coverage_color(coverage),
                        ).classes("w-full")

    def _on_checkbox(path: str, default_gt: str):
        def handler(e):
            new_sel = state.get_selected_sources()
            if e.value:
                new_sel[path] = new_sel.get(path, default_gt)
            else:
                new_sel.pop(path, None)
            state.set_selected_sources(new_sel)
            summary.refresh()
        return handler

    def _on_gt(path: str):
        def handler(e):
            new_sel = state.get_selected_sources()
            if path in new_sel:
                new_sel[path] = e.value
                state.set_selected_sources(new_sel)
                summary.refresh()
        return handler

    def _on_delete(path: str):
        def handler():
            config_manager.remove_data_source(path)
            new_sel = state.get_selected_sources()
            new_sel.pop(path, None)
            state.set_selected_sources(new_sel)
            content.refresh()
            summary.refresh()
            ui.notify("Source supprimée.", type="info")
        return handler

    def _open_add_dialog():
        existing = {s["path"] for s in config_manager.get_data_sources()}
        detected = [p for p in config_manager.auto_scan_sources()
                    if p not in existing]

        with ui.dialog() as dialog, ui.card().classes("w-[600px] q-pa-md gap-2"):
            ui.label("Ajouter une source").classes("text-h6")

            # Sources détectées automatiquement
            ui.label("Sources détectées automatiquement :"
                     ).classes("text-caption text-grey q-mt-sm")
            if detected:
                for p in detected:
                    name = Path(p).name.replace("_", " ").title()
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.column().classes("gap-0"):
                            ui.label(name).classes("text-body2")
                            ui.label(p).classes("text-caption text-grey")
                        ui.button(
                            "+ Ajouter",
                            on_click=_add_detected(p, dialog),
                        ).props("dense color=primary")
            else:
                ui.label("Aucune nouvelle source détectée."
                         ).classes("text-caption text-grey")

            ui.separator()

            # Ajout manuel
            ui.label("Ajout manuel :").classes("text-caption text-grey")
            path_input = ui.input("Chemin du dossier").classes("w-full")
            corpus_input = ui.input("Corpus", value="Autre").classes("w-full")

            def add_manual():
                p = (path_input.value or "").strip()
                if not p:
                    ui.notify("Veuillez saisir un chemin de dossier.",
                              type="warning")
                    return
                corpus = (corpus_input.value or "").strip() or "Autre"
                config_manager.add_data_source(p, label="", corpus=corpus)
                content.refresh()
                summary.refresh()
                dialog.close()
                ui.notify("Source ajoutée.", type="positive")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Annuler", on_click=dialog.close).props("flat")
                ui.button("Ajouter", on_click=add_manual).props("color=primary")

        dialog.open()

    def _add_detected(path: str, dialog):
        def handler():
            config_manager.add_data_source(path)
            content.refresh()
            summary.refresh()
            dialog.close()
            ui.notify("Source ajoutée.", type="positive")
            _open_add_dialog()  # rafraîchit la liste des sources détectées
        return handler

    # ── Rendu initial ──
    content()
    ui.separator()
    summary()
