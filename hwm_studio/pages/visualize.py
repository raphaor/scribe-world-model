"""Page « Visualisation » de HWM Studio.

Construit un formulaire qui assemble la commande CLI pour
``visualize.py`` (mode validation sur split seedé, ou mode fichier/dossier
sans split), en affiche un aperçu en temps réel, puis la lance via
:mod:`hwm_studio.runner` tout en diffusant la sortie dans un panneau de
logs. Le script ouvre une fenêtre matplotlib externe (navigation page par
page) : un avertissement le rappelle à l'utilisateur.

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

# hwm_studio/pages/visualize.py → parent³ = racine du projet (scribe-world-model/)
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
    """Scanne les fichiers ``.pkl`` à la racine (modèles de langue n-gram)."""
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
    """Construit la page Visualisation."""

    # --- Découverte des versions, checkpoints et LM --------------------------
    versions = _get_model_versions()
    pt_files = _scan_pt_files()
    pkl_files = _scan_pkl_files()

    # S'assurer que la version par défaut existe dans la liste connue
    _default_version = "v17"
    if _default_version not in versions and versions:
        _default_version = versions[-1]

    # --- Dictionnaire d'état du formulaire -----------------------------------
    form: dict = {
        "mode": "validation",
        "model": "hwm_v17.pt",
        "model_version": _default_version,
        "split": "val",
        "alto_file": "",
        "max_samples": 50,
        "sort_by_cer": False,
        "top_n": None,
        "seed": 42,
        "beam_search": False,
        "lm_path": "",
        "beam_width": 20,
        "lm_weight": 0.3,
        "workers": 0,
        "no_gt": False,
        "min_frames_per_char": 0.0,
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

    def _refresh_preview():
        """Rafraîchit uniquement l'aperçu commande (best-effort)."""
        try:
            command_preview.refresh()
        except Exception:
            pass

    # --- Sections rafraîchissables (déclarées avant les handlers) ------------
    @ui.refreshable
    def sources_bar():
        """Affiche le nombre de sources sélectionnées (mode validation)."""
        if form.get("mode") != "validation":
            return
        try:
            sources = state.get_selected_sources()
        except Exception:
            sources = {}
        n_total = len(sources)
        n_gt = sum(1 for m in sources.values() if m == "gt")
        with ui.row().classes("items-center q-mb-sm"):
            if n_total == 0:
                ui.icon("warning", color="warning")
                ui.label(
                    "Aucune source sélectionnée — le script utilisera "
                    "config.ALTO_DIRS par défaut."
                ).classes("text-warning")
            else:
                gt_str = f" · {n_gt} en GT" if n_gt else ""
                ui.icon("folder_open", color="info")
                ui.label(
                    f"Sources sélectionnées : {n_total} dossier(s){gt_str}"
                )
            ui.link("Sources →", "/?page=sources").classes("q-ml-sm")

    @ui.refreshable
    def mode_section():
        """Champs dépendant du mode : split (validation) ou chemin (fichier)."""
        if form.get("mode") == "validation":
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("Split").classes("w-28")
                ui.select(
                    options=["val", "train"],
                    value=form.get("split", "val"),
                    on_change=on_change("split"),
                ).classes("w-40").tooltip(
                    "Split évalué en mode validation (defaut: val)."
                )
        else:
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("Chemin").classes("w-28")
                ui.input(
                    value=form.get("alto_file", ""),
                    placeholder="page.xml ou /media/rapha/B4FC7F32FC7EEE4C/OCR_genealogie/Alto/<dir>",
                    on_change=on_change("alto_file"),
                ).classes("w-72").tooltip(
                    "Fichier .xml ALTO unique ou répertoire de .xml. "
                    "Pas de split — toutes les lignes sont affichées."
                )
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("").classes("w-28")  # alignement
                ui.label(
                    "Astuce : --no-gt prédit même les lignes sans "
                    "transcription."
                ).classes("text-caption text-grey")

    @ui.refreshable
    def top_n_section():
        """Sous-formulaire Top N (visible uniquement si tri par CER actif)."""
        if not form.get("sort_by_cer"):
            return
        with ui.row().classes("items-center q-mb-sm q-ml-md"):
            ui.label("Top N").classes("w-28")
            ui.number(
                value=form.get("top_n"),
                min=1, step=1, clearable=True,
                placeholder="tous",
                on_change=on_change("top_n"),
            ).classes("w-32").tooltip(
                "Avec --sort-by-cer, n'afficher que les N pires."
            )

    @ui.refreshable
    def beam_section():
        """Sous-formulaire beam search (visible si beam search actif)."""
        if not form.get("beam_search"):
            return
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("LM (.pkl)").classes("w-28")
            ui.select(
                options=pkl_files,
                value=form.get("lm_path") or None,
                with_input=True, clearable=True,
                on_change=on_change("lm_path"),
                label="(n-gram LM, optionnel)",
            ).classes("w-72")
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Beam width").classes("w-28")
            ui.number(
                value=form["beam_width"], min=1, step=1,
                on_change=on_change("beam_width"),
            ).classes("w-24").tooltip("Largeur du beam (defaut: 20).")
            ui.label("LM weight").classes("q-ml-md w-24")
            ui.number(
                value=form["lm_weight"], step=0.05,
                on_change=on_change("lm_weight"),
            ).classes("w-24").tooltip(
                "Poids du LM dans le score du beam (defaut: 0.3)."
            )
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Workers").classes("w-28")
            ui.number(
                value=form["workers"], min=0, step=1,
                on_change=on_change("workers"),
            ).classes("w-24").tooltip(
                "Worker processes (0=sequentiel)."
            )

    # --- Handlers des switches (rafraîchissent les sous-sections) ------------
    def on_mode_change(e):
        form["mode"] = e.value
        try:
            sources_bar.refresh()
        except Exception:
            pass
        try:
            mode_section.refresh()
        except Exception:
            pass
        _refresh_preview()

    def on_sort_toggle(e):
        form["sort_by_cer"] = e.value
        try:
            top_n_section.refresh()
        except Exception:
            pass
        _refresh_preview()

    def on_beam_toggle(e):
        form["beam_search"] = e.value
        try:
            beam_section.refresh()
        except Exception:
            pass
        _refresh_preview()

    # --- Construction de la liste d'arguments CLI ----------------------------
    def build_args(dirs=None) -> list[str]:
        """Lit le formulaire et construit la liste d'arguments pour visualize.py.

        Seules les valeurs non-défaut sont émises.

        Args:
            dirs: liste de dossiers ALTO résolus (mode validation). Si
                ``None``, utilise :func:`state.get_resolved_alto_dirs`.
        """
        args: list[str] = []

        # --- Checkpoint (uniquement si ≠ défaut) ---
        ckpt = str(form.get("model") or "").strip()
        if ckpt and ckpt != "hwm_v17.pt":
            args += ["--model", ckpt]

        # --- Version (toujours, choices requis) ---
        args += ["--model-version", str(form.get("model_version") or "v17")]

        # --- Mode validation : --alto-dirs + --split ---
        if form.get("mode") == "validation":
            if dirs is None:
                try:
                    dirs = state.get_resolved_alto_dirs()
                except Exception:
                    dirs = []
            if dirs:
                args += ["--alto-dirs", *dirs]
            split = form.get("split") or "val"
            if split != "val":
                args += ["--split", split]
        else:
            # --- Mode fichier/dossier : --alto-file ---
            path = str(form.get("alto_file") or "").strip()
            if path:
                args += ["--alto-file", path]

        # --- Max samples (uniquement si ≠ 50) ---
        try:
            ms = int(form.get("max_samples") or 50)
            if ms != 50:
                args += ["--max-samples", str(ms)]
        except (TypeError, ValueError):
            pass

        # --- Sort by CER + Top N ---
        if form.get("sort_by_cer"):
            args.append("--sort-by-cer")
            try:
                tn = form.get("top_n")
                if tn is not None:
                    tn = int(tn)
                    if tn > 0:
                        args += ["--top-n", str(tn)]
            except (TypeError, ValueError):
                pass

        # --- Seed (uniquement si ≠ 42) ---
        try:
            seed = int(form.get("seed") or 42)
            if seed != 42:
                args += ["--seed", str(seed)]
        except (TypeError, ValueError):
            pass

        # --- Min frames per char (uniquement si ≠ 0.0) ---
        try:
            mfc = float(form.get("min_frames_per_char") or 0.0)
            if mfc:
                args += ["--min-frames-per-char", str(mfc)]
        except (TypeError, ValueError):
            pass

        # --- Beam search + sous-paramètres ---
        if form.get("beam_search"):
            args.append("--beam-search")
            lm = str(form.get("lm_path") or "").strip()
            if lm:
                args += ["--lm-path", lm]
            try:
                bw = int(form.get("beam_width") or 20)
                if bw != 20:
                    args += ["--beam-width", str(bw)]
            except (TypeError, ValueError):
                pass
            try:
                lw = float(form.get("lm_weight") or 0.3)
                if lw != 0.3:
                    args += ["--lm-weight", str(lw)]
            except (TypeError, ValueError):
                pass
            try:
                w = int(form.get("workers") or 0)
                if w:
                    args += ["--workers", str(w)]
            except (TypeError, ValueError):
                pass

        # --- No GT ---
        if form.get("no_gt"):
            args.append("--no-gt")

        return args

    # ===================================================================== #
    #  Mise en page
    # ===================================================================== #

    ui.label("Visualisation").classes("text-h4 q-mb-md")

    # --- Sélection du mode + avertissement matplotlib ------------------------
    with ui.row().classes("items-center q-mb-md"):
        ui.toggle(
            {"validation": "Validation", "file": "Fichier / Dossier"},
            value=form["mode"],
            on_change=on_mode_change,
        ).classes("q-mr-md")

    ui.label(
        "⚠️ Cette commande ouvre une fenêtre matplotlib externe."
    ).classes("text-warning q-mb-md")

    # --- Carte formulaire ----------------------------------------------------
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Paramètres du modèle").classes("text-h6 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Checkpoint").classes("w-28")
            ui.select(
                options=pt_files,
                value=form["model"],
                with_input=True,
                on_change=on_change("model"),
                label="(fichiers .pt à la racine)",
            ).classes("w-72")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Version").classes("w-28")
            ui.select(
                options=versions,
                value=form["model_version"],
                on_change=on_change("model_version"),
            ).classes("w-40")

        ui.separator().classes("q-my-sm")
        ui.label("Données").classes("text-subtitle1 q-mb-sm")

        # Barre sources + section mode-dépendante
        sources_bar()
        mode_section()

        ui.separator().classes("q-my-sm")
        ui.label("Échantillonnage").classes("text-subtitle1 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Max samples").classes("w-28")
            ui.number(
                value=form["max_samples"], min=1, step=1,
                on_change=on_change("max_samples"),
            ).classes("w-40").tooltip(
                "Nombre d'échantillons à afficher (defaut: 50)."
            )
            ui.label("Seed").classes("q-ml-md w-16")
            ui.number(
                value=form["seed"], step=1,
                on_change=on_change("seed"),
            ).classes("w-24").tooltip(
                "Seed pour le sous-échantillonnage (defaut: 42)."
            )

        with ui.row().classes("items-center q-mb-sm"):
            ui.switch(
                "Trier par CER",
                value=form["sort_by_cer"],
                on_change=on_sort_toggle,
            ).tooltip(
                "Trier par CER descendant (pires cas d'abord)."
            )
        top_n_section()

        ui.separator().classes("q-my-sm")
        ui.label("Beam search (optionnel)").classes("text-subtitle1 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.switch(
                "Beam search",
                value=form["beam_search"],
                on_change=on_beam_toggle,
            ).tooltip(
                "Activer le CTC beam search (en plus du greedy)."
            )
        beam_section()

        ui.separator().classes("q-my-sm")
        ui.label("Autres").classes("text-subtitle1 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.switch(
                "No GT",
                value=form["no_gt"],
                on_change=on_change("no_gt"),
            ).tooltip(
                "Prédire toutes les lignes même sans ground truth "
                "(nécessite --alto-file)."
            )

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Min frames/char").classes("w-28")
            ui.number(
                value=form["min_frames_per_char"], min=0, step=0.1,
                on_change=on_change("min_frames_per_char"),
            ).classes("w-24").tooltip(
                "Doit correspondre à la valeur d'entraînement (defaut: 0.0)."
            )

    # --- Aperçu de la commande -----------------------------------------------
    ui.label("Commande :").classes("text-subtitle1 q-mb-xs")

    @ui.refreshable
    def command_preview():
        """Affiche la commande CLI reconstituée à partir du formulaire."""
        try:
            args = build_args()
            cmd = runner.build_command_str("visualize.py", args)
        except Exception:
            cmd = "python visualize.py  (erreur de construction)"
        ui.code(cmd, language="bash").classes("w-full")

    command_preview()

    # --- Boutons d'action ----------------------------------------------------
    def copy_command():
        """Copie la commande actuelle dans le presse-papiers."""
        try:
            cmd = runner.build_command_str("visualize.py", build_args())
            ui.run_javascript(
                f"navigator.clipboard.writeText({json.dumps(cmd)})"
            )
            ui.notify("Commande copiée", type="info")
        except Exception as exc:
            ui.notify(f"Échec de la copie : {exc}", type="negative")

    async def on_launch():
        """Lance visualize.py via le runner."""
        if runner.is_running():
            ui.notify("Un processus est déjà en cours", type="negative")
            return

        # Validation minimale selon le mode
        if form.get("mode") == "validation":
            try:
                dirs = state.get_resolved_alto_dirs()
            except Exception:
                dirs = []
            if not dirs:
                ui.notify(
                    "Aucune source sélectionnée — config.ALTO_DIRS sera "
                    "utilisé",
                    type="warning",
                )
        else:
            path = str(form.get("alto_file") or "").strip()
            if not path:
                ui.notify(
                    "Chemin --alto-file requis en mode Fichier / Dossier",
                    type="warning",
                )
                return
            dirs = None

        args = build_args(dirs)

        log_panel.clear()
        ui.notify("Démarrage…", type="info")
        try:
            await runner.run_script(
                "visualize.py", args,
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
