"""Tests du rendu graphique pygame (game/renderer.py).

Ces tests tournent « headless » (sans fenêtre) grâce au pilote vidéo factice
de SDL : ils vérifient seulement que le rendu s'exécute sans erreur et produit
une surface aux bonnes dimensions. Si pygame n'est pas installé, ils sont
ignorés (skip).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Pilote SDL factice : pas de vraie fenêtre, utilisable en CI.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pygame = pytest.importorskip("pygame")

from game.renderer import BoardRenderer  # noqa: E402  (après importorskip, voulu)


@pytest.fixture(autouse=True)
def _pygame_session():
    """Initialise puis ferme pygame autour de chaque test."""
    pygame.init()
    yield
    pygame.quit()


def test_renderer_dimensions() -> None:
    """Les dimensions calculées sont cohérentes et positives."""
    renderer = BoardRenderer(size=4, cell=100, gap=10, margin=15, header=60)
    assert renderer.width == 15 * 2 + 4 * 100 + 3 * 10
    assert renderer.height == 60 + 15 * 2 + 4 * 100 + 3 * 10


def test_draw_runs_without_error() -> None:
    """Dessiner une grille remplit la surface sans lever d'exception."""
    renderer = BoardRenderer()
    surface = pygame.Surface((renderer.width, renderer.height))
    grid = np.array(
        [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [0, 0, 2, 2],
        ]
    )
    renderer.draw(surface, grid, title="Partie test", subtitle="score 1234")

    # La surface a bien la taille attendue.
    assert surface.get_size() == (renderer.width, renderer.height)
    # Au moins un pixel a été peint (la surface n'est pas restée transparente).
    assert surface.get_at((0, 0))[:3] != (0, 0, 0) or surface.get_at((5, 5))[:3] != (0, 0, 0)
