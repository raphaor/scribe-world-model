"""Page « Évaluation CER » de HWM Studio.

Construit un formulaire qui assemble la commande CLI pour
``recognize.py``, en affiche un aperçu en temps réel, puis la lance via
:mod:`hwm_studio.runner` tout en diffusant la sortie dans un panneau de
logs en temps réel.

La page est volontairement défensive : une valeur de formulaire
manquante ou une erreur d'import ne doit jamais faire planter le GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from hwm_studio import runner, state


# --------------------------------------------------------------------------- #
#  Racine du projet & helpers de découverte
# --------------------------------------------------------------------------- #

# hwm_studio/pages/evaluate.py → parent³ = racine du projet (scribe-world-model/)
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


def _get_model_versions() -> list[str]:
    """Retourne la liste des versions de modèle connues.

    Tente d'importer ``model_registry`` (ce qui déclenche l'import de
    ``torch``) ; en cas d'échec, retombe sur une liste codée en dur.
    """
    try:
        from model_registry import known_versions
        versions = known_versions()
        if versions:
            return versions
    except Exception:
        pass
    return [
        "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
        "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18",
    ]


def _scan_pt_files() -> list[str]:
    """Scanne les fichiers ``.pt`` à la racine du projet (autocomplete)."""
    try:
        return sorted(
            p.name for p in _PROJECT_ROOT.glob("*.pt") if p.is_file()
        )
    except Exception:
        return []


def _scan_pkl_files() -> list[str]:
    """Scanne les fichiers ``.pkl`` à la racine du projet (modèles de langue)."""
    try:
        return sorted(
            p.name for p in _PROJECT_ROOT.glob("*.pkl") if p.is_file()
        )
    except Exception:
        return []


# --------------------------------------------------------------------------- #
#  Construction de la page
# --------------------------------------------------------------------------- #

def build():
    """Construit la page Évaluation CER."""

    # --- Découverte des versions, checkpoints et modèles de langue -----------
    versions = _get_model_versions()
    pt_files = _scan_pt_files()
    pkl_files = _scan_pkl_files()

    # Checkpoint par défaut : "hwm_v4.pt" (défaut CLI) s'il existe, sinon
    # le premier .pt trouvé.
    _default_model = "hwm_v4.pt"
    if _default_model not in pt_files and pt_files:
        _default_model = pt_files[0]

    # Version par défaut cohérente avec le checkpoint si possible
    _default_version = "v5"
    if _default_model.startswith("hwm_v") and _default_version in versions:
        _default_version = "v5"

    # --- Dictionnaire d'état du formulaire -----------------------------------
    form: dict = {
        "model": _default_model,
        "model_version": _default_version,
        "split": "val",
        "batch_size": 32,
        "max_samples": None,
        "beam_search": False,
        "lm_path": None,
        "beam_width": 20,
        "lm_weight": 0.3,
        "workers": 0,
        "compare": False,
    }

    # --- Helper : mise à jour d'un champ + rafraîchissement -------------------
    def on_change(key):
        """Crée un callback on_change qui stocke la valeur et rafraîchit."""
        def handler(e):
            form[key] = e.value
            try:
                command_preview.refresh()
            except Exception:
                pass
        return handler

    # --- Construction de la liste d'arguments CLI ----------------------------
    def build_args(dirs=None) -> list[str]:
        """Lit le formulaire et construit la liste d'arguments pour recognize.py.

        N'inclut que les valeurs non-défaut. Inclut toujours ``--alto-dirs``
        avec les dossiers résolus.

        Args:
            dirs: liste de dossiers ALTO résolus. Si ``None``, utilise
                :func:`state.get_resolved_alto_dirs`.
        """
        if dirs is None:
            try:
                dirs = state.get_resolved_alto_dirs()
            except Exception:
                dirs = []

        args: list[str] = []

        # --- Modèle (toujours requis) ---
        model = form.get("model")
        if model:
            args += ["--model", str(model)]

        # --- Version ---
        args += ["--model-version", str(form.get("model_version") or "v5")]

        # --- Split (défaut: val) ---
        split = form.get("split") or "val"
        if split != "val":
            args += ["--split", split]

        # --- Batch size (défaut: 32) ---
        try:
            bs = int(form.get("batch_size") or 32)
            if bs != 32:
                args += ["--batch-size", str(bs)]
        except (TypeError, ValueError):
            pass

        # --- Max samples (optionnel) ---
        try:
            ms = form.get("max_samples")
            if ms is not None:
                ms_int = int(ms)
                if ms_int > 0:
                    args += ["--max-samples", str(ms_int)]
        except (TypeError, ValueError):
            pass

        # --- Compare (flag) — implique beam search ---
        compare = bool(form.get("compare"))

        # --- Beam search (flag) ---
        beam = bool(form.get("beam_search")) or compare
        if beam:
            args.append("--beam-search")

        # --- Options beam search (uniquement si activé) ---
        if beam:
            # LM path (optionnel)
            lm = form.get("lm_path")
            if lm:
                args += ["--lm-path", str(lm)]

            # Beam width (défaut: 20)
            try:
                bw = int(form.get("beam_width") or 20)
                if bw != 20:
                    args += ["--beam-width", str(bw)]
            except (TypeError, ValueError):
                pass

            # LM weight (défaut: 0.3)
            try:
                lw = float(form.get("lm_weight") or 0.3)
                if lw != 0.3:
                    args += ["--lm-weight", str(lw)]
            except (TypeError, ValueError):
                pass

            # Workers (défaut: 0)
            try:
                w = int(form.get("workers") or 0)
                if w:
                    args += ["--workers", str(w)]
            except (TypeError, ValueError):
                pass

        # --- Compare (flag) ---
        if compare:
            args.append("--compare")

        # --- Dossiers ALTO (toujours présents si résolus) ---
        if dirs:
            args += ["--alto-dirs", *dirs]

        return args

    # ===================================================================== #
    #  Mise en page
    # ===================================================================== #

    ui.label("Évaluation CER").classes("text-h4 q-mb-md")

    # --- Barre des sources sélectionnées -------------------------------------
    try:
        sources = state.get_selected_sources()
    except Exception:
        sources = {}
    n_total = len(sources)
    n_gt = sum(1 for m in sources.values() if m == "gt")
    with ui.row().classes("items-center q-mb-md"):
        if n_total == 0:
            ui.icon("warning", color="warning")
            ui.label("Aucune source sélectionnée.").classes("text-warning")
        else:
            gt_str = f" · {n_gt} en GT" if n_gt else ""
            ui.label(f"Sources sélectionnées : {n_total} dossier(s){gt_str}")
        ui.button(
            "Sources →", on_click=lambda: ui.navigate.to("/?page=sources")
        ).props("flat dense")

    # --- Formulaire principal ------------------------------------------------
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Paramètres").classes("text-h6 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Checkpoint").classes("w-32")
            ui.select(
                options=pt_files,
                value=form["model"],
                with_input=True,
                on_change=on_change("model"),
                label="(fichiers .pt à la racine)",
            ).classes("w-72").tooltip(
                "Checkpoint du modèle à évaluer."
            )

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Version").classes("w-32")
            ui.select(
                options=versions,
                value=form["model_version"],
                on_change=on_change("model_version"),
            ).classes("w-32").tooltip(
                "Version d'architecture du modèle."
            )
            ui.label("Split").classes("q-ml-md w-16")
            ui.toggle(
                {x: x for x in ["val", "train", "all"]},
                value=form["split"],
                on_change=on_change("split"),
            ).classes("q-ml-xs")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Batch size").classes("w-32")
            ui.number(
                value=form["batch_size"], min=1, step=1,
                on_change=on_change("batch_size"),
            ).classes("w-32")
            ui.label("Max samples").classes("q-ml-md w-28")
            ui.number(
                value=form["max_samples"], min=0, step=1,
                on_change=on_change("max_samples"),
            ).classes("w-32").tooltip(
                "Sous-ensemble aléatoire_seedé (plus rapide pour le beam "
                "search). Vide = dataset complet."
            )

        ui.separator().classes("q-my-md")

        # --- Beam search + options conditionnelles ---
        beam_switch = ui.switch(
            "Beam search",
            value=form["beam_search"],
            on_change=on_change("beam_search"),
        ).tooltip(
            "Utiliser le CTC beam search au lieu du décodage greedy."
        )

        # Conteneur des options beam (visibilité liée au switch)
        beam_options = ui.column().classes("q-pl-lg gap-1")
        beam_options.bind_visibility_from(beam_switch, "value")

        with beam_options:
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("LM (.pkl)").classes("w-32")
                ui.select(
                    options=pkl_files,
                    value=form["lm_path"],
                    with_input=True, clearable=True,
                    on_change=on_change("lm_path"),
                    label="(optionnel)",
                ).classes("w-72").tooltip(
                    "Modèle de langue char n-gramme (issu de train_lm.py). "
                    "Si fourni, biaise le beam search."
                )

            with ui.row().classes("items-center q-mb-sm"):
                ui.label("Beam width").classes("w-32")
                ui.number(
                    value=form["beam_width"], min=1, step=1,
                    on_change=on_change("beam_width"),
                ).classes("w-32")
                ui.label("LM weight").classes("q-ml-md w-24")
                ui.number(
                    value=form["lm_weight"], min=0, max=1, step=0.05,
                    on_change=on_change("lm_weight"),
                ).classes("w-32").tooltip(
                    "Poids d'interpolation LM. 0 = beam CTC pur, "
                    "1 = dominé par le LM."
                )

            with ui.row().classes("items-center"):
                ui.label("Workers").classes("w-32")
                ui.number(
                    value=form["workers"], min=0, step=1,
                    on_change=on_change("workers"),
                ).classes("w-32").tooltip(
                    "Processus workers pour le beam search "
                    "(0 = séquentiel)."
                )

        # --- Compare mode ---
        def on_compare_change(e):
            """Active automatiquement le beam search si compare est activé."""
            form["compare"] = e.value
            if e.value:
                beam_switch.value = True
                form["beam_search"] = True
            try:
                command_preview.refresh()
            except Exception:
                pass

        ui.switch(
            "Mode comparaison (greedy + beam)",
            value=form["compare"],
            on_change=on_compare_change,
        ).tooltip(
            "Lance à la fois le décodage greedy ET beam search, "
            "puis affiche les deux CER."
        )

    # --- Aperçu de la commande -----------------------------------------------
    ui.label("Commande :").classes("text-subtitle1 q-mb-xs")

    @ui.refreshable
    def command_preview():
        """Affiche la commande CLI reconstituée à partir du formulaire."""
        try:
            args = build_args()
            cmd = runner.build_command_str("recognize.py", args)
        except Exception:
            cmd = "python recognize.py  (erreur de construction)"
        ui.code(cmd, language="bash").classes("w-full")

    command_preview()

    # --- Boutons d'action ----------------------------------------------------
    def copy_command():
        """Copie la commande actuelle dans le presse-papiers."""
        try:
            cmd = runner.build_command_str("recognize.py", build_args())
            ui.run_javascript(
                f"navigator.clipboard.writeText({json.dumps(cmd)})"
            )
            ui.notify("Commande copiée", type="info")
        except Exception as exc:
            ui.notify(f"Échec de la copie : {exc}", type="negative")

    async def on_launch():
        """Lance l'évaluation via le runner."""
        if runner.is_running():
            ui.notify("Un processus est déjà en cours", type="negative")
            return
        try:
            dirs = state.get_resolved_alto_dirs()
        except Exception:
            dirs = []
        if not dirs:
            ui.notify("Aucune source sélectionnée", type="warning")
            return
        args = build_args(dirs)

        log_panel.clear()
        ui.notify("Démarrage…", type="info")
        try:
            await runner.run_script(
                "recognize.py", args,
                on_output=lambda line: log_panel.push(line),
                on_done=lambda rc: ui.notify(
                    f"Terminé (code {rc})",
                    type="positive" if rc == 0 else "negative"),
            )
        except Exception as exc:
            ui.notify(f"Erreur : {exc}", type="negative")

    def on_stop():
        """Demande l'arrêt du processus actif."""
        if runner.stop():
            ui.notify("Arrêt demandé", type="warning")
        else:
            ui.notify("Aucun processus actif", type="info")

    with ui.row().classes("items-center q-mt-sm q-mb-md"):
        ui.button("Copier", icon="content_copy", on_click=copy_command)
        ui.space()
        ui.button("Lancer", icon="rocket_launch", color="positive",
                  on_click=on_launch)
        ui.button("Stop", color="negative", on_click=on_stop)

    # --- Panneau de logs -----------------------------------------------------
    ui.label("Sortie :").classes("text-subtitle1 q-mb-xs")
    log_panel = ui.log(max_lines=200).classes("w-full h-64")
