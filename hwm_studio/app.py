"""Coquille principale de l'application HWM Studio (NiceGUI).

Point d'entrée du GUI. Construit la mise en page commune — tiroir latéral
de navigation, en-tête, barre d'état en bas — et route vers les pages
individuelles via le paramètre de requête ``?page=<clé>``.

Usage::

    python -m hwm_studio.app
"""

from __future__ import annotations

import importlib

from fastapi import Request
from nicegui import ui

from hwm_studio import runner


# --------------------------------------------------------------------------- #
#  Catalogue des pages
# --------------------------------------------------------------------------- #
# Chaque entrée : (libellé affiché, icône Material, module relatif à hwm_studio)
PAGES: dict[str, tuple[str, str, str]] = {
    'dashboard':  ('Dashboard',          'dashboard',      'pages.dashboard'),
    'sources':    ('Sources de données', 'folder_open',    'pages.sources'),
    'train':      ('Entraînement',       'model_training', 'pages.train'),
    'evaluate':   ('Évaluation CER',     'fact_check',     'pages.evaluate'),
    'visualize':  ('Visualisation',      'image',          'pages.visualize'),
    'annotate':   ('Annotation',         'edit_note',      'pages.annotate'),
    'models':     ('Modèles',            'storage',        'pages.models'),
    'train_lm':   ('Langue (LM)',        'translate',      'pages.train_lm'),
    'help':       ('Aide',               'help_outline',   'pages.help'),
}

# Page affichée quand aucune clé n'est fournie ou qu'elle est invalide.
_PAGE_DEFAUT: str = 'dashboard'


# --------------------------------------------------------------------------- #
#  Chargement dynamique des pages
# --------------------------------------------------------------------------- #

def load_page(page_key: str) -> None:
    """Importe et construit la page demandée.

    Si le module de la page n'existe pas encore ou lève une exception, un
    emplacement « Bientôt disponible » est affiché à la place — l'appli
    ne plante jamais.
    """
    label, icon, module_path = PAGES.get(page_key, PAGES[_PAGE_DEFAUT])
    try:
        module = importlib.import_module(f'hwm_studio.{module_path}')
        module.build()
    except Exception as exc:  # noqa: BLE001 — on veut tout rattraper ici
        _placeholder(label, icon, exc)


def _placeholder(label: str, icon: str, exc: Exception | None = None) -> None:
    """Affiche un emplacement réservé centré pour une page non disponible."""
    with ui.column().classes('absolute-center items-center text-center gap-2'):
        ui.icon(icon, size='80px', color='primary')
        ui.label(label).classes('text-h5')
        ui.label('Bientôt disponible').classes('text-grey')
        if exc is not None:
            ui.label(str(exc)).classes('text-caption text-grey-6')


# --------------------------------------------------------------------------- #
#  Barre d'état (bas de page)
# --------------------------------------------------------------------------- #

@ui.refreshable
def status_bar() -> None:
    """Construit le contenu de la barre d'état selon l'état du runner.

    * Processus actif : point rouge + commande en cours + bouton « Stop ».
    * Au repos        : point vert + « Prêt ».
    """
    if runner.is_running():
        commande = runner.get_active_command() or 'En cours…'
        ui.icon('circle', color='red', size='14px')
        ui.label(commande).classes('flex-grow ellipsis')
        ui.button('Stop', on_click=stop_process).props('color=negative dense')
    else:
        ui.icon('circle', color='green', size='14px')
        ui.label('Prêt')


def stop_process() -> None:
    """Demande l'arrêt du processus actif puis rafraîchit la barre d'état."""
    runner.stop()
    status_bar.refresh()


def _update_status() -> None:
    """Rafraîchit la barre d'état (appelée périodiquement par le timer)."""
    status_bar.refresh()


# --------------------------------------------------------------------------- #
#  Navigation
# --------------------------------------------------------------------------- #

def go(page_key: str) -> None:
    """Navigue vers une page via le paramètre de requête ``?page=<clé>``."""
    ui.navigate.to(f'/?page={page_key}')


# --------------------------------------------------------------------------- #
#  Page principale
# --------------------------------------------------------------------------- #

@ui.page('/')
def main_page(request: Request) -> None:
    """Construit la mise en page complète pour la connexion courante."""
    page_key = request.query_params.get('page', _PAGE_DEFAUT)
    if page_key not in PAGES:
        page_key = _PAGE_DEFAUT

    # Thème sombre activé par défaut
    ui.dark_mode().enable()

    # En-tête
    with ui.header().classes('items-center q-px-md'):
        ui.icon('history_edu', size='28px')
        ui.label('HWM Studio').classes('text-h6 q-ml-xs')

    # Tiroir latéral de navigation
    with ui.left_drawer(top_corner=True, bottom_corner=True) \
            .props('width=250') \
            .classes('column gap-1 q-pa-sm'):
        ui.label('Navigation').classes('text-caption text-grey q-mb-xs')
        for key, (label, icon, _) in PAGES.items():
            actif = key == page_key
            bouton = ui.button(label, icon=icon, on_click=lambda k=key: go(k))
            bouton.props('flat align=left')
            bouton.classes('w-full justify-start')
            if actif:
                bouton.classes('bg-primary text-white')

    # Barre d'état en bas de page
    with ui.footer().classes('items-center'):
        with ui.row().classes('w-full items-center q-px-md q-py-xs'):
            status_bar()

    # Zone de contenu principale
    load_page(page_key)

    # Rafraîchissement périodique de la barre d'état (toutes les 2 s)
    ui.timer(2.0, _update_status)


# --------------------------------------------------------------------------- #
#  Point d'entrée
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    ui.run(host='127.0.0.1', port=8080, title='HWM Studio', reload=False)
