"""Page « Entraînement LM » de HWM Studio.

Construit un formulaire qui assemble la commande CLI pour
``train_lm.py`` (modèle de langue n-gramme de caractères pour le beam
search), en affiche un aperçu en temps réel, puis la lance via
:mod:`hwm_studio.runner` tout en diffusant la sortie dans un panneau de
logs en temps réel.

La page est volontairement défensive : une valeur de formulaire
manquante ou une erreur d'import ne doit jamais faire planter le GUI.
"""

from __future__ import annotations

import json

from nicegui import ui

from hwm_studio import runner, state


# --------------------------------------------------------------------------- #
#  Helpers de découverte
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
#  Construction de la page
# --------------------------------------------------------------------------- #

def build():
    """Construit la page Entraînement LM."""

    # --- Découverte des versions --------------------------------------------
    versions = _get_model_versions()
    _default_version = "v17"
    if _default_version not in versions:
        _default_version = versions[0] if versions else "v17"

    # --- Dictionnaire d'état du formulaire ----------------------------------
    form: dict = {
        "model_version": _default_version,
        "output": "char_8gram.pkl",
        "order": 8,
        "min_frames_per_char": 0.0,
    }

    # --- Helper : mise à jour d'un champ + rafraîchissement -----------------
    def on_change(key):
        """Crée un callback on_change qui stocke la valeur et rafraîchit."""
        def handler(e):
            form[key] = e.value
            try:
                command_preview.refresh()
            except Exception:
                pass
        return handler

    # --- Construction de la liste d'arguments CLI ---------------------------
    def build_args(dirs=None) -> list[str]:
        """Lit le formulaire et construit la liste d'arguments pour train_lm.py.

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

        # Version du modèle (pour img_height et cnn_width_stride)
        args += ["--model-version", form.get("model_version") or "v17"]

        # Fichier de sortie .pkl
        out = str(form.get("output") or "").strip()
        if out:
            args += ["--output", out]

        # Ordre du n-gramme
        try:
            order = int(form.get("order") or 8)
            args += ["--order", str(order)]
        except (TypeError, ValueError):
            args += ["--order", "8"]

        # Min frames per char (uniquement si != 0.0 — valeur par défaut du script)
        try:
            mfc = float(form.get("min_frames_per_char") or 0.0)
            if mfc != 0.0:
                args += ["--min-frames-per-char", str(mfc)]
        except (TypeError, ValueError):
            pass

        # Dossiers ALTO (toujours présents si sources sélectionnées)
        if dirs:
            args += ["--alto-dirs", *dirs]

        return args

    # ===================================================================== #
    #  Mise en page
    # ===================================================================== #

    ui.label("Entraînement modèle de langue (n-gramme)").classes("text-h4 q-mb-md")

    ui.label(
        "Génère un modèle n-gramme de caractères à partir des transcriptions "
        "d'entraînement. Utilisable ensuite avec le beam search "
        "(évaluation, visualisation, annotation)."
    ).classes("text-body2 q-mb-md")

    # --- Barre des sources sélectionnées ------------------------------------
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
        ui.link("Sources →", "/?page=sources").classes("q-ml-sm")

    # --- Formulaire ---------------------------------------------------------
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Paramètres").classes("text-h6 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Version du modèle").classes("w-40")
            ui.select(
                options=versions,
                value=form["model_version"],
                on_change=on_change("model_version"),
            ).classes("w-40").tooltip(
                "Version du modèle — détermine img_height et cnn_width_stride "
                "(doit correspondre au modèle utilisé pour la reconnaissance)."
            )

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Fichier de sortie").classes("w-40")
            ui.input(
                value=form["output"],
                placeholder="char_8gram.pkl",
                on_change=on_change("output"),
            ).classes("w-72").tooltip(
                "Chemin du fichier .pkl qui contiendra le modèle n-gramme."
            )

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Ordre du n-gramme").classes("w-40")
            ui.number(
                value=form["order"], min=2, max=12, step=1,
                on_change=on_change("order"),
            ).classes("w-40").tooltip(
                "Ordre du modèle n-gramme (ex. 8 = 8-gramme de caractères)."
            )

        with ui.row().classes("items-center"):
            ui.label("Min frames/char").classes("w-40")
            ui.number(
                value=form["min_frames_per_char"], min=0.0, step=0.5,
                on_change=on_change("min_frames_per_char"),
            ).classes("w-40").tooltip(
                "Doit correspondre à la valeur utilisée lors de l'entraînement "
                "du modèle (filtre les lignes dont le rapport frames/caractère "
                "est trop faible). 0.0 = aucune exclusion. "
                "Ex. v18 R8 utilisait 1.0."
            )

    # --- Aperçu de la commande ----------------------------------------------
    ui.label("Commande :").classes("text-subtitle1 q-mb-xs")

    @ui.refreshable
    def command_preview():
        """Affiche la commande CLI reconstituée à partir du formulaire."""
        try:
            args = build_args()
            cmd = runner.build_command_str("train_lm.py", args)
        except Exception:
            cmd = "python train_lm.py  (erreur de construction)"
        ui.code(cmd, language="bash").classes("w-full")

    command_preview()

    # --- Boutons d'action ---------------------------------------------------
    def copy_command():
        """Copie la commande actuelle dans le presse-papiers."""
        try:
            cmd = runner.build_command_str("train_lm.py", build_args())
            ui.run_javascript(
                f"navigator.clipboard.writeText({json.dumps(cmd)})"
            )
            ui.notify("Commande copiée", type="info")
        except Exception as exc:
            ui.notify(f"Échec de la copie : {exc}", type="negative")

    async def on_launch():
        """Lance l'entraînement du LM via le runner."""
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
                "train_lm.py", args,
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

    # --- Panneau de logs ----------------------------------------------------
    ui.label("Sortie :").classes("text-subtitle1 q-mb-xs")
    log_panel = ui.log(max_lines=200).classes("w-full h-64")
