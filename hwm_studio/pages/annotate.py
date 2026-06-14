"""Page « Annotation » de HWM Studio.

Construit un formulaire qui assemble la commande CLI pour le script
racine ``annotate.py`` (annotation/review interactif de ground truth
ALTO via matplotlib), en affiche un aperçu en temps réel, puis la lance
via :mod:`hwm_studio.runner` tout en diffusant la sortie dans un panneau
de logs en temps réel.

La page est volontairement défensive : une valeur de formulaire
manquante ou une erreur d'import ne doit jamais faire planter le GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from hwm_studio import runner


# --------------------------------------------------------------------------- #
#  Racine du projet & helpers de découverte
# --------------------------------------------------------------------------- #

# hwm_studio/pages/annotate.py → parent³ = racine du projet (scribe-world-model/)
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
    """Scanne les fichiers ``hwm_*.pt`` à la racine du projet (autocomplete)."""
    try:
        return sorted(
            p.name for p in _PROJECT_ROOT.glob("hwm_*.pt") if p.is_file()
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
    """Construit la page Annotation."""

    # --- Découverte des versions, checkpoints et LM --------------------------
    versions = _get_model_versions()
    pt_files = _scan_pt_files()
    pkl_files = _scan_pkl_files()

    _default_model = "hwm_v17.pt"
    if _default_model not in pt_files and pt_files:
        _default_model = pt_files[-1]

    # --- Dictionnaire d'état du formulaire -----------------------------------
    form: dict = {
        "alto_path": "",
        "model": _default_model,
        "model_version": "v17",
        "mode": "annotate",
        "beam_search": False,
        "lm_path": "",
        "beam_width": 20,
        "lm_weight": 0.3,
        "workers": 0,
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
    def build_args() -> list[str]:
        """Lit le formulaire et construit la liste d'arguments pour annotate.py."""
        args: list[str] = []

        # --- Chemin ALTO (obligatoire) ---
        alto = str(form.get("alto_path") or "").strip()
        if alto:
            args += ["--alto", alto]

        # --- Checkpoint (uniquement si ≠ défaut) ---
        model = str(form.get("model") or "").strip()
        if model and model != "hwm_v17.pt":
            args += ["--model", model]

        # --- Version ---
        args += ["--model-version", form.get("model_version") or "v17"]

        # --- Mode (uniquement si ≠ défaut "annotate") ---
        mode = form.get("mode") or "annotate"
        if mode != "annotate":
            args += ["--mode", mode]

        # --- Beam search ---
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

        return args

    # ===================================================================== #
    #  Mise en page
    # ===================================================================== #

    ui.label("Annotation de ground truth").classes("text-h4 q-mb-md")

    # --- Bandeau d'information -----------------------------------------------
    with ui.row().classes("items-center q-mb-md"):
        ui.icon("info", color="info")
        ui.label(
            "L'annotation crée des fichiers `_gt.xml` à côté des originaux. "
            "Les originaux ne sont jamais modifiés."
        ).classes("text-grey")

    # --- Formulaire ----------------------------------------------------------
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Paramètres").classes("text-h6 q-mb-sm")

        # Chemin ALTO (obligatoire)
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Chemin ALTO *").classes("w-28")
            ui.input(
                value=form["alto_path"],
                placeholder="Fichier .xml ou répertoire",
                on_change=on_change("alto_path"),
            ).classes("w-72").tooltip(
                "Fichier .xml unique ou répertoire contenant des fichiers .xml ALTO"
            )

        # Checkpoint + version
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Checkpoint").classes("w-28")
            ui.select(
                options=pt_files,
                value=form["model"] or None,
                with_input=True, clearable=True,
                on_change=on_change("model"),
                label="(fichiers hwm_*.pt à la racine)",
            ).classes("w-72")
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Version").classes("w-28")
            ui.select(
                options=versions,
                value=form["model_version"],
                on_change=on_change("model_version"),
            ).classes("w-40")

        # Mode
        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Mode").classes("w-28")
            ui.toggle_button(
                {"annotate": "Annoter (sans GT)", "review": "Revoir (avec GT)"},
                value=form["mode"],
                on_change=on_change("mode"),
            )

        ui.separator().classes("q-my-sm")

        # --- Options beam search (conditionnelles, révélées par le switch) -----
        @ui.refreshable
        def beam_options():
            if not form.get("beam_search"):
                return
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("Modèle de langue").classes("w-40")
                ui.select(
                    options=pkl_files,
                    value=form["lm_path"] or None,
                    with_input=True, clearable=True,
                    on_change=on_change("lm_path"),
                    label="(.pkl à la racine)",
                ).classes("w-60")
            with ui.row().classes("items-center q-mb-sm"):
                ui.label("Beam width").classes("w-40")
                ui.number(
                    value=form["beam_width"], min=1, step=1,
                    on_change=on_change("beam_width"),
                ).classes("w-24")
                ui.label("LM weight").classes("q-ml-md w-24")
                ui.number(
                    value=form["lm_weight"], min=0, step=0.05,
                    on_change=on_change("lm_weight"),
                ).classes("w-24")
            with ui.row().classes("items-center"):
                ui.label("Workers").classes("w-40")
                ui.number(
                    value=form["workers"], min=0, step=1,
                    on_change=on_change("workers"),
                ).classes("w-24").tooltip(
                    "Processus workers pour le beam search (0 = séquentiel)"
                )

        def on_beam_change(e):
            """Active/désactive le beam search et rafraîchit les options."""
            form["beam_search"] = e.value
            try:
                beam_options.refresh()
                command_preview.refresh()
            except Exception:
                pass

        # Switch beam search — déclenche l'affichage des options avancées
        with ui.row().classes("items-center q-mb-sm"):
            ui.switch(
                "Beam search",
                value=form["beam_search"],
                on_change=on_beam_change,
            ).tooltip("Activer le CTC beam search pour les prédictions")

        beam_options()

    # --- Avertissement fenêtre interactive -----------------------------------
    with ui.row().classes("items-center q-mb-md"):
        ui.icon("warning", color="warning")
        ui.label(
            "⚠️ Cette commande ouvre une fenêtre matplotlib interactive."
        ).classes("text-warning")

    # --- Aperçu de la commande -----------------------------------------------
    ui.label("Commande :").classes("text-subtitle1 q-mb-xs")

    @ui.refreshable
    def command_preview():
        """Affiche la commande CLI reconstituée à partir du formulaire."""
        try:
            args = build_args()
            cmd = runner.build_command_str("annotate.py", args)
        except Exception:
            cmd = "python annotate.py  (erreur de construction)"
        ui.code(cmd, language="bash").classes("w-full")

    command_preview()

    # --- Boutons d'action ----------------------------------------------------
    def copy_command():
        """Copie la commande actuelle dans le presse-papiers."""
        try:
            cmd = runner.build_command_str("annotate.py", build_args())
            ui.run_javascript(
                f"navigator.clipboard.writeText({json.dumps(cmd)})"
            )
            ui.notify("Commande copiée", type="info")
        except Exception as exc:
            ui.notify(f"Échec de la copie : {exc}", type="negative")

    async def on_launch():
        """Lance annotate.py via le runner."""
        alto = str(form.get("alto_path") or "").strip()
        if not alto:
            ui.notify("Le chemin ALTO est obligatoire (--alto)", type="negative")
            return
        if runner.is_running():
            ui.notify("Un processus est déjà en cours", type="negative")
            return
        args = build_args()

        log_panel.clear()
        ui.notify("Démarrage…", type="info")
        try:
            await runner.run_script(
                "annotate.py", args,
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
