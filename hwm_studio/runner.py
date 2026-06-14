"""
Gestionnaire de sous-processus pour HWM Studio.

Lance des scripts Python (train.py, recognize.py, etc.) en tant que
sous-processus, capture stdout/stderr en temps réel et fournit des
fonctions d'arrêt. Un seul processus peut tourner à la fois (verrou global).
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable

# --- Racine du projet ---
# parent de hwm_studio/ (donc parent du dossier contenant ce fichier)
_PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)

# --- État global (un seul processus à la fois) ---
_active_process: asyncio.subprocess.Process | None = None
_active_command: str = ""
_active_process_id: str | None = None

# Compteur pour générer des identifiants de processus lisibles
_process_counter: int = 0


# --------------------------------------------------------------------------- #
#  Utilitaires
# --------------------------------------------------------------------------- #

def _next_process_id() -> str:
    """Génère un identifiant unique et lisible pour un processus."""
    global _process_counter
    _process_counter += 1
    return f"proc_{_process_counter}"


def get_project_root() -> str:
    """Retourne le chemin racine du projet (parent de hwm_studio/)."""
    return _PROJECT_ROOT


def get_venv_python() -> str:
    """
    Retourne le chemin de l'exécutable Python du venv.

    Sur Windows, cherche ``venv/Scripts/python.exe`` à la racine du projet.
    Si introuvable, revient à :data:`sys.executable`.
    """
    candidate = os.path.join(_PROJECT_ROOT, "venv", "Scripts", "python.exe")
    if os.path.isfile(candidate):
        return candidate
    return sys.executable


def is_running() -> bool:
    """Retourne ``True`` si un sous-processus est actuellement actif."""
    return _active_process is not None


def get_active_command() -> str | None:
    """Retourne la commande lisible du processus actif, ou ``None``."""
    if _active_process is not None:
        return _active_command or None
    return None


def build_command_str(script: str, args: list[str]) -> str:
    """
    Construit une chaîne de commande lisible à partir du nom du script
    et de ses arguments.

    Les flags suivis d'une valeur sont regroupés::

        build_command_str("train.py", ["--epochs", "50"])
            -> "python train.py --epochs 50"

        build_command_str("train.py", ["--no-jepa"])
            -> "python train.py --no-jepa"
    """
    parts: list[str] = [f"python {script}"]
    i = 0
    while i < len(args):
        arg = args[i]
        # Un flag (--xxx) suivi d'une valeur (ne commence pas par '-')
        if arg.startswith("-") and i + 1 < len(args) and not args[i + 1].startswith("-"):
            parts.append(f"{arg} {args[i + 1]}")
            i += 2
        else:
            # Flag seul ou valeur positionnelle
            parts.append(arg)
            i += 1
    return " ".join(parts)


# --------------------------------------------------------------------------- #
#  Lancement et suivi de processus
# --------------------------------------------------------------------------- #

async def run_script(
    script: str,
    args: list[str],
    on_output: Callable[[str], None],
    on_done: Callable[[int], None],
) -> str:
    """
    Lance un script Python en sous-processus et capture sa sortie en continu.

    Args:
        script: nom du fichier (ex. ``"train.py"``), relatif à la racine
            du projet.
        args: arguments CLI (ex. ``["--model-version", "v17", "--epochs", "50"]``).
        on_output: callback appelé pour chaque ligne de stdout/stderr.
        on_done: callback appelé avec le code de retour à la fin du processus.

    Returns:
        L'identifiant du processus.

    Raises:
        RuntimeError: si un processus est déjà en cours.
    """
    global _active_process, _active_command, _active_process_id

    if is_running():
        raise RuntimeError(f"Un processus est déjà en cours: {_active_command}")

    python_exe = get_venv_python()
    project_root = get_project_root()
    script_path = os.path.join(project_root, script)

    # Commande lisible pour l'affichage dans l'UI
    _active_command = build_command_str(script, args)
    process_id = _next_process_id()
    _active_process_id = process_id

    # Environnement: sortie non tamponnée + encodage UTF-8 forcé
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    # Commande complète exécutée
    cmd = [python_exe, script_path, *args]

    # Lancement du sous-processus (stderr fusionné dans stdout)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_root,
            env=env,
        )
    except Exception:
        # Échec du lancement — réinitialiser l'état avant de propager
        _active_command = ""
        _active_process_id = None
        raise

    _active_process = proc

    try:
        # Lecture en continu des lignes de sortie
        while True:
            try:
                # Timeout court: rester réactif même sans sortie,
                # pour vérifier régulièrement l'état du processus.
                line_bytes = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=0.5
                )
            except asyncio.TimeoutError:
                # Aucune ligne pendant ce délai — le processus tourne
                # peut-être encore silencieusement, ou vient de se terminer.
                if proc.returncode is not None:
                    break
                continue

            if not line_bytes:
                # EOF — le flux de sortie est fermé, le processus est fini
                break

            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
            on_output(line)

        # Récupérer le code de retour définitif
        returncode = await proc.wait()
    finally:
        _active_process = None
        _active_command = ""
        _active_process_id = None

    on_done(returncode)
    return process_id


# --------------------------------------------------------------------------- #
#  Arrêt de processus
# --------------------------------------------------------------------------- #

async def _finalize_stop(proc: asyncio.subprocess.Process) -> None:
    """
    Attend la fin du processus après :meth:`terminate`.

    Si le processus ne se termine pas dans les 5 secondes, force un
    :meth:`kill`. Exécuté en arrière-plan via :meth:`loop.create_task`.
    """
    try:
        await asyncio.wait_for(proc.wait(), timeout=5)
        return  # Le processus s'est terminé proprement après terminate()
    except asyncio.TimeoutError:
        pass

    # Toujours en vie après 5 s — kill() forcé
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await proc.wait()
    except Exception:
        pass


def stop() -> bool:
    """
    Arrête le processus actif.

    Fonction **synchrone** — peut être appelée directement depuis un
    gestionnaire d'événement NiceGUI.

    Procédure:

    1. ``terminate()`` (SIGTERM / TerminateProcess sur Windows).
    2. Si toujours en vie après 5 secondes: ``kill()``.

    Returns:
        ``True`` si l'arrêt a été demandé, ``False`` si aucun processus
        n'était actif (ou déjà terminé).
    """
    if _active_process is None:
        return False

    proc = _active_process
    if proc.returncode is not None:
        # Le processus est déjà terminé
        return False

    # Étape 1: terminate() — demande d'arrêt
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        pass

    # Programmer l'escalade kill() après 5 s si le processus résiste
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Pas de boucle disponible — terminate() a quand même été envoyé
        return True

    loop.create_task(_finalize_stop(proc))

    return True
