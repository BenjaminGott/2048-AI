"""Rendu texte/ANSI d'une grille 2048.

Module partagé par tous les outils qui affichent une grille dans le terminal
(`play.py`, `scripts/watch_games.py`, `scripts/replay_game.py`). Centraliser
ce code évite de dupliquer la palette de couleurs et la logique d'encadrement.

Aucune logique de jeu ici : on ne manipule que des grilles (tableaux 4×4).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# --- Palette de couleurs (codes ANSI 256) ----------------------------------
# Pour chaque valeur de tuile : (couleur de fond, couleur de texte).
# Au-delà de 2048, on réutilise la dernière couleur.
RESET = "\x1b[0m"
TILE_COLORS: dict[int, tuple[int, int]] = {
    0: (236, 244),  # case vide : gris foncé
    2: (255, 236),  # blanc cassé, texte foncé
    4: (223, 236),  # beige
    8: (215, 235),  # orange clair
    16: (208, 231),  # orange
    32: (202, 231),  # orange-rouge
    64: (196, 231),  # rouge
    128: (227, 236),  # jaune clair
    256: (220, 236),  # jaune
    512: (214, 236),  # jaune-or
    1024: (190, 16),  # vert-jaune
    2048: (46, 16),  # vert vif
}
MAX_COLOR_KEY = 2048


def ansi_cell(value: int, cell_width: int = 6) -> str:
    """Représente une case colorée de largeur fixe.

    Args:
        value (int): valeur de la tuile (0 = case vide).
        cell_width (int): largeur intérieure de la case.

    Returns:
        str: chaîne colorée (codes ANSI), de largeur visible `cell_width`.
    """
    bg, fg = TILE_COLORS.get(min(value, MAX_COLOR_KEY), TILE_COLORS[MAX_COLOR_KEY])
    text = "" if value == 0 else str(value)
    return f"\x1b[48;5;{bg}m\x1b[38;5;{fg}m{text.rjust(cell_width)}{RESET}"


def board_block(
    grid: np.ndarray | Sequence[Sequence[int]],
    header_lines: Sequence[str] = (),
    cell_width: int = 6,
) -> list[str]:
    """Construit les lignes d'une grille encadrée (avec en-tête optionnel).

    Toutes les lignes renvoyées ont la même largeur visible, ce qui permet de
    poser plusieurs blocs côte à côte via `side_by_side`.

    Args:
        grid: grille 2D (tableau numpy ou liste de listes).
        header_lines: lignes de texte affichées au-dessus de la grille.
        cell_width: largeur d'une case.

    Returns:
        list[str]: lignes prêtes à être imprimées.
    """
    grid = np.asarray(grid)
    size = grid.shape[0]

    top = "┌" + "┬".join(["─" * cell_width] * size) + "┐"
    mid = "├" + "┼".join(["─" * cell_width] * size) + "┤"
    bot = "└" + "┴".join(["─" * cell_width] * size) + "┘"
    width = len(top)  # largeur visible commune (top n'a aucun code ANSI)

    lines: list[str] = [h.ljust(width) for h in header_lines]
    lines.append(top)
    for r in range(size):
        cells = "│".join(ansi_cell(int(v), cell_width) for v in grid[r])
        lines.append("│" + cells + "│")
        lines.append(mid if r < size - 1 else bot)
    return lines


def side_by_side(blocks: Sequence[list[str]], sep: str = "   ") -> list[str]:
    """Pose plusieurs blocs côte à côte (même hauteur attendue).

    Args:
        blocks: liste de blocs (chacun étant une liste de lignes).
        sep: séparateur horizontal entre deux blocs.

    Returns:
        list[str]: lignes combinées.
    """
    height = max(len(b) for b in blocks)
    # On complète les blocs plus courts avec des lignes vides de même largeur.
    padded: list[list[str]] = []
    for block in blocks:
        width = len(block[0]) if block else 0
        padded.append(block + [" " * width] * (height - len(block)))
    return [sep.join(block[i] for block in padded) for i in range(height)]


def grid_of_blocks(blocks: Sequence[list[str]], cols: int, col_sep: str = "   ") -> str:
    """Dispose des blocs en grille (plusieurs par ligne) et renvoie le texte.

    Args:
        blocks: blocs à disposer.
        cols: nombre de blocs par ligne.
        col_sep: séparateur horizontal entre blocs.

    Returns:
        str: l'affichage complet, prêt à être imprimé.
    """
    out: list[str] = []
    for start in range(0, len(blocks), cols):
        row = blocks[start : start + cols]
        out.extend(side_by_side(row, sep=col_sep))
        out.append("")  # ligne vide entre deux rangées
    return "\n".join(out)
