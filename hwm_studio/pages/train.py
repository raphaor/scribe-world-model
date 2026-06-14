"""Page « Entraînement » de HWM Studio.

Construit un formulaire qui assemble la commande CLI pour ``train.py``,
en affiche un aperçu en temps réel, puis la lance via
:mod:`hwm_studio.runner` tout en diffusant la sortie dans un panneau de
logs en temps réel.

La page est volontairement défensive : une valeur de formulaire
manquante ou une erreur d'import ne doit jamais faire planter le GUI.
"""

from __future__ import annotations

import json
from pathlib import Path

from nicegui import ui

from hwm_studio import runner, state, config_manager


# --------------------------------------------------------------------------- #
#  Racine du projet & helpers de découverte
# --------------------------------------------------------------------------- #

# hwm_studio/pages/train.py → parent³ = racine du projet (scribe-world-model/)
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


def _default_save_path(version: str) -> str:
    """Retourne le chemin de checkpoint par défaut pour une version."""
    try:
        from model_registry import get_spec
        sp = get_spec(version).save_path
        if sp:
            return sp
    except Exception:
        pass
    return f"hwm_{version}.pt"


# --------------------------------------------------------------------------- #
#  Construction de la page
# --------------------------------------------------------------------------- #

def build():
    """Construit la page Entraînement."""

    # --- Découverte des versions, checkpoints et préférences -----------------
    versions = _get_model_versions()
    pt_files = _scan_pt_files()
    try:
        recent: dict = config_manager.load_config().get("recent_training", {}) or {}
    except Exception:
        recent = {}

    # S'assurer que la version par défaut existe dans la liste connue
    _default_version = recent.get("model_version", "v17")
    if _default_version not in versions:
        _default_version = "v17"

    # --- Dictionnaire d'état du formulaire -----------------------------------
    form: dict = {
        "model_version": _default_version,
        "mode": recent.get("mode", "mixed"),
        "epochs": recent.get("epochs", 30),
        "batch_size": recent.get("batch_size", 32),
        "lr": recent.get("lr", 1e-3),
        "checkpoint": recent.get("checkpoint", ""),
        "save_path": recent.get("save_path", ""),
        "no_augment": recent.get("no_augment", False),
        "no_amp": recent.get("no_amp", False),
        "no_bucket": recent.get("no_bucket", False),
        "no_jepa": recent.get("no_jepa", False),
        "constant_lr": recent.get("constant_lr", False),
        "phase_restart": recent.get("phase_restart", False),
        "grad_checkpoint": recent.get("grad_checkpoint", False),
        "encoder_lr_mult": recent.get("encoder_lr_mult", 0.1),
        "warmup_epochs": recent.get("warmup_epochs", 0),
        "freeze_encoder_epochs": recent.get("freeze_encoder_epochs", 0),
        "lambda_pred": "",          # vide = auto (config par défaut du modèle)
        "lambda_sigreg": "",        # vide = auto
        "min_frames_per_char": recent.get("min_frames_per_char", 0.0),
        "num_workers": recent.get("num_workers", 0),
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
        """Lit le formulaire et construit la liste d'arguments pour train.py.

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
        v = form.get("model_version") or "v17"
        args += ["--model-version", v]
        args += ["--mode", form.get("mode") or "mixed"]

        # --- Paramètres essentiels numériques ---
        try:
            args += ["--epochs", str(int(form.get("epochs") or 30))]
        except (TypeError, ValueError):
            args += ["--epochs", "30"]
        try:
            args += ["--batch-size", str(int(form.get("batch_size") or 32))]
        except (TypeError, ValueError):
            args += ["--batch-size", "32"]
        try:
            args += ["--lr", str(float(form.get("lr") or 1e-3))]
        except (TypeError, ValueError):
            args += ["--lr", "1e-3"]

        # --- Checkpoint (optionnel) ---
        ckpt = form.get("checkpoint")
        if ckpt:
            args += ["--checkpoint", str(ckpt)]

        # --- Save path (uniquement si différent du défaut) ---
        sp = str(form.get("save_path") or "").strip()
        if sp and sp != _default_save_path(v):
            args += ["--save-path", sp]

        # --- Flags booléens ---
        for flag, key in [
            ("--no-augment", "no_augment"),
            ("--no-amp", "no_amp"),
            ("--no-bucket", "no_bucket"),
            ("--no-jepa", "no_jepa"),
            ("--constant-lr", "constant_lr"),
            ("--phase-restart", "phase_restart"),
            ("--grad-checkpoint", "grad_checkpoint"),
        ]:
            if form.get(key):
                args.append(flag)

        # --- Params numériques avancés (ajoutés seulement si ≠ défaut) ---
        try:
            enc = float(form.get("encoder_lr_mult") or 0.1)
            if enc != 0.1:
                args += ["--encoder-lr-mult", str(enc)]
        except (TypeError, ValueError):
            pass
        try:
            warm = int(form.get("warmup_epochs") or 0)
            if warm:
                args += ["--warmup-epochs", str(warm)]
        except (TypeError, ValueError):
            pass
        try:
            freeze = int(form.get("freeze_encoder_epochs") or 0)
            if freeze:
                args += ["--freeze-encoder-epochs", str(freeze)]
        except (TypeError, ValueError):
            pass
        try:
            mfc = float(form.get("min_frames_per_char") or 0.0)
            if mfc:
                args += ["--min-frames-per-char", str(mfc)]
        except (TypeError, ValueError):
            pass
        try:
            nw = int(form.get("num_workers") or 0)
            if nw:
                args += ["--num-workers", str(nw)]
        except (TypeError, ValueError):
            pass

        # --- Lambda (uniquement si valeur explicite, 0 est valide) ---
        for flag, key in [("--lambda-pred", "lambda_pred"),
                          ("--lambda-sigreg", "lambda_sigreg")]:
            val = str(form.get(key) or "").strip()
            if val:
                try:
                    float(val)          # validation seulement
                    args += [flag, val]
                except ValueError:
                    pass

        # --- Dossiers ALTO (toujours présents si sources sélectionnées) ---
        if dirs:
            args += ["--alto-dirs", *dirs]

        return args

    # ===================================================================== #
    #  Mise en page
    # ===================================================================== #

    ui.label("Entraînement").classes("text-h4 q-mb-md")

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
        ui.link("Sources →", "/?page=sources").classes("q-ml-sm")

    # --- Paramètres essentiels -----------------------------------------------
    with ui.card().classes("w-full q-mb-md"):
        ui.label("Paramètres essentiels").classes("text-h6 q-mb-sm")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Version").classes("w-28")
            ui.select(
                options=versions,
                value=form["model_version"],
                on_change=on_change("model_version"),
            ).classes("w-40")
            ui.label("Mode").classes("q-ml-md w-16")
            ui.select(
                options=["mixed", "full", "adapt"],
                value=form["mode"],
                on_change=on_change("mode"),
            ).classes("w-32")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Epochs").classes("w-28")
            ui.number(
                value=form["epochs"], min=1, step=1,
                on_change=on_change("epochs"),
            ).classes("w-40")
            ui.label("Batch").classes("q-ml-md w-16")
            ui.number(
                value=form["batch_size"], min=1, step=1,
                on_change=on_change("batch_size"),
            ).classes("w-32")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("LR").classes("w-28")
            ui.number(
                value=form["lr"], step=0.0001,
                on_change=on_change("lr"),
            ).classes("w-40")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Checkpoint").classes("w-28")
            ui.select(
                options=pt_files,
                value=form["checkpoint"] or None,
                with_input=True, clearable=True,
                on_change=on_change("checkpoint"),
                label="(fichiers .pt à la racine)",
            ).classes("w-72")

        with ui.row().classes("items-center"):
            ui.label("Save path").classes("w-28")
            ui.input(
                value=form["save_path"],
                placeholder=f"défaut: {_default_save_path(form['model_version'])}",
                on_change=on_change("save_path"),
            ).classes("w-72").tooltip(
                "Chemin de sauvegarde du checkpoint. "
                "Laisser vide pour utiliser le défaut de la version."
            )

    # --- Paramètres avancés --------------------------------------------------
    with ui.expansion("Paramètres avancés", icon="tune").classes("w-full q-mb-md"):
        with ui.row().classes("q-mb-sm"):
            ui.checkbox("No augment", value=form["no_augment"],
                        on_change=on_change("no_augment")).tooltip(
                "Désactiver l'augmentation d'images")
            ui.checkbox("No AMP", value=form["no_amp"],
                        on_change=on_change("no_amp")).tooltip(
                "Désactiver AMP (float16)")
            ui.checkbox("No bucket", value=form["no_bucket"],
                        on_change=on_change("no_bucket")).tooltip(
                "Désactiver le bucketing de largeur")
            ui.checkbox("No JEPA", value=form["no_jepa"],
                        on_change=on_change("no_jepa")).tooltip(
                "Désactiver la branche JEPA / SSL")
        with ui.row().classes("q-mb-sm"):
            ui.checkbox("Constant LR", value=form["constant_lr"],
                        on_change=on_change("constant_lr")).tooltip(
                "Désactiver le decay cosinus du LR")
            ui.checkbox("Phase restart", value=form["phase_restart"],
                        on_change=on_change("phase_restart")).tooltip(
                "Forcer le reset de phase (compteur, optimiseur, scheduler)")
            ui.checkbox("Grad checkpoint", value=form["grad_checkpoint"],
                        on_change=on_change("grad_checkpoint")).tooltip(
                "Gradient checkpointing (~30 % de VRAM gagnée)")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Encoder LR mult").classes("w-32")
            ui.number(value=form["encoder_lr_mult"], step=0.05,
                      on_change=on_change("encoder_lr_mult")).classes("w-24")
            ui.label("Warmup").classes("q-ml-md w-16")
            ui.number(value=form["warmup_epochs"], min=0, step=1,
                      on_change=on_change("warmup_epochs")).classes("w-20")
            ui.label("Freeze").classes("q-ml-md w-16")
            ui.number(value=form["freeze_encoder_epochs"], min=0, step=1,
                      on_change=on_change("freeze_encoder_epochs")).classes("w-20")

        with ui.row().classes("items-center q-mb-sm"):
            ui.label("Lambda pred").classes("w-32")
            ui.input(value=form["lambda_pred"], placeholder="auto",
                     on_change=on_change("lambda_pred")).classes("w-24").tooltip(
                "Poids de la loss JEPA. Vide = défaut config.")
            ui.label("Lambda sigreg").classes("q-ml-md w-24")
            ui.input(value=form["lambda_sigreg"], placeholder="auto",
                     on_change=on_change("lambda_sigreg")).classes("w-24").tooltip(
                "Poids SIGReg. Vide = défaut config.")

        with ui.row().classes("items-center"):
            ui.label("Min frames/char").classes("w-32")
            ui.number(value=form["min_frames_per_char"], min=0, step=0.1,
                      on_change=on_change("min_frames_per_char")).classes("w-24")
            ui.label("Workers").classes("q-ml-md w-16")
            ui.number(value=form["num_workers"], min=0, step=1,
                      on_change=on_change("num_workers")).classes("w-20")

    # --- Aperçu de la commande -----------------------------------------------
    ui.label("Commande :").classes("text-subtitle1 q-mb-xs")

    @ui.refreshable
    def command_preview():
        """Affiche la commande CLI reconstituée à partir du formulaire."""
        try:
            args = build_args()
            cmd = runner.build_command_str("train.py", args)
        except Exception:
            cmd = "python train.py  (erreur de construction)"
        ui.code(cmd, language="bash").classes("w-full")

    command_preview()

    # --- Boutons d'action ----------------------------------------------------
    def copy_command():
        """Copie la commande actuelle dans le presse-papiers."""
        try:
            cmd = runner.build_command_str("train.py", build_args())
            ui.run_javascript(
                f"navigator.clipboard.writeText({json.dumps(cmd)})"
            )
            ui.notify("Commande copiée", type="info")
        except Exception as exc:
            ui.notify(f"Échec de la copie : {exc}", type="negative")

    async def on_launch():
        """Lance l'entraînement via le runner."""
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

        # Mémoriser les paramètres pour la prochaine fois
        try:
            config_manager.save_recent_training({
                "model_version": form.get("model_version"),
                "mode": form.get("mode"),
                "epochs": form.get("epochs"),
                "batch_size": form.get("batch_size"),
                "lr": form.get("lr"),
                "checkpoint": form.get("checkpoint"),
                "save_path": form.get("save_path"),
                "no_augment": form.get("no_augment"),
                "no_amp": form.get("no_amp"),
                "no_bucket": form.get("no_bucket"),
                "no_jepa": form.get("no_jepa"),
                "constant_lr": form.get("constant_lr"),
                "phase_restart": form.get("phase_restart"),
                "grad_checkpoint": form.get("grad_checkpoint"),
                "encoder_lr_mult": form.get("encoder_lr_mult"),
                "warmup_epochs": form.get("warmup_epochs"),
                "freeze_encoder_epochs": form.get("freeze_encoder_epochs"),
                "min_frames_per_char": form.get("min_frames_per_char"),
                "num_workers": form.get("num_workers"),
            })
        except Exception:
            pass

        log_panel.clear()
        ui.notify("Démarrage…", type="info")
        try:
            await runner.run_script(
                "train.py", args,
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
