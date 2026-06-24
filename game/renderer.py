"""Rendu graphique (pygame) d'une grille 2048.

Ce module dessine une grille 2048 dans une surface pygame, avec la palette de
couleurs classique du jeu. Il ne contient aucune logique de jeu : on lui passe
simplement une grille (tableau 4×4) à afficher. Il est utilisé par le
visualiseur graphique `scripts/replay_gui.py`.

pygame est une dépendance optionnelle (rendu visuel uniquement) : ce module
n'est importé que par les outils graphiques, jamais par le cœur du jeu.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pygame

# --- Palette de couleurs classique du 2048 (RGB) ---------------------------
BG_COLOR = (187, 173, 160)  # fond de la grille
EMPTY_CELL = (205, 193, 180)  # case vide
HEADER_COLOR = (250, 248, 239)  # bandeau d'en-tête
TEXT_DARK = (119, 110, 101)  # texte sur tuiles claires (2, 4)
TEXT_LIGHT = (249, 246, 242)  # texte sur tuiles foncées

TILE_RGB: dict[int, tuple[int, int, int]] = {
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
MAX_TILE_KEY = 2048


class BoardRenderer:
    """Dessine une grille 2048 (et un en-tête) dans une surface pygame."""

    def __init__(
        self, size: int = 4, cell: int = 110, gap: int = 12, margin: int = 16, header: int = 70
    ) -> None:
        """Configure les dimensions du rendu.

        Args:
            size (int): nombre de cases par côté (4 pour le 2048 standard).
            cell (int): côté d'une case en pixels.
            gap (int): espace entre deux cases.
            margin (int): marge autour de la grille.
            header (int): hauteur du bandeau d'en-tête (titre + score).
        """
        self.size = size
        self.cell = cell
        self.gap = gap
        self.margin = margin
        self.header = header
        self._fonts: dict[int, pygame.font.Font] = {}

    @property
    def width(self) -> int:
        """Largeur totale en pixels de la surface attendue."""
        return self.margin * 2 + self.size * self.cell + (self.size - 1) * self.gap

    @property
    def height(self) -> int:
        """Hauteur totale en pixels de la surface attendue."""
        return self.header + self.margin * 2 + self.size * self.cell + (self.size - 1) * self.gap

    def _font(self, pixels: int) -> pygame.font.Font:
        """Renvoie (en le mettant en cache) une police d'une taille donnée.

        On utilise la police par défaut intégrée à pygame (`Font(None, ...)`) :
        contrairement à `SysFont`, elle est toujours disponible, y compris en
        environnement sans polices système (CI, rendu headless).
        """
        if pixels not in self._fonts:
            self._fonts[pixels] = pygame.font.Font(None, pixels)
        return self._fonts[pixels]

    def draw(
        self,
        surface: pygame.Surface,
        grid: np.ndarray | Sequence[Sequence[int]],
        title: str = "",
        subtitle: str = "",
    ) -> None:
        """Dessine la grille et son en-tête sur la surface fournie.

        Args:
            surface (pygame.Surface): surface cible (taille width × height).
            grid: grille 2D à afficher.
            title (str): texte affiché à gauche de l'en-tête.
            subtitle (str): texte affiché à droite de l'en-tête (ex: score).
        """
        grid = np.asarray(grid)
        surface.fill(BG_COLOR)

        # --- En-tête (titre à gauche, sous-titre à droite) ---
        pygame.draw.rect(surface, HEADER_COLOR, (0, 0, self.width, self.header))
        font_header = self._font(28)
        if title:
            surface.blit(
                font_header.render(title, True, TEXT_DARK), (self.margin, self.header // 4)
            )
        if subtitle:
            img = font_header.render(subtitle, True, TEXT_DARK)
            surface.blit(img, (self.width - self.margin - img.get_width(), self.header // 4))

        # --- Cases ---
        for r in range(self.size):
            for c in range(self.size):
                value = int(grid[r][c])
                x = self.margin + c * (self.cell + self.gap)
                y = self.header + self.margin + r * (self.cell + self.gap)
                color = (
                    EMPTY_CELL
                    if value == 0
                    else TILE_RGB.get(min(value, MAX_TILE_KEY), TILE_RGB[MAX_TILE_KEY])
                )
                pygame.draw.rect(surface, color, (x, y, self.cell, self.cell), border_radius=8)
                if value:
                    self._draw_value(surface, value, x, y)

    def _draw_value(self, surface: pygame.Surface, value: int, x: int, y: int) -> None:
        """Dessine le nombre centré dans une case (taille adaptée à sa longueur)."""
        text = str(value)
        # Plus le nombre est long, plus on réduit la police pour qu'il tienne.
        base = self.cell // 2
        font_size = max(16, base - max(0, len(text) - 2) * (base // 5))
        color = TEXT_DARK if value <= 4 else TEXT_LIGHT
        img = self._font(font_size).render(text, True, color)
        rect = img.get_rect(center=(x + self.cell // 2, y + self.cell // 2))
        surface.blit(img, rect)
