"""Logique pure du jeu 2048.

Ce module contient uniquement la mécanique du jeu (grille, glissement,
fusion, score, fin de partie). Il ne contient AUCUNE logique d'affichage :
la séparation des responsabilités permet de réutiliser cette classe aussi
bien dans le script terminal `play.py` que dans l'environnement Gymnasium.

Convention des directions :
    0 = haut, 1 = bas, 2 = gauche, 3 = droite
"""

from __future__ import annotations

import numpy as np

# --- Constantes globales ---------------------------------------------------

GRID_SIZE: int = 4  # grille carrée 4x4
NEW_TILE_VALUES: tuple[int, int] = (2, 4)  # valeurs possibles d'une nouvelle tuile
PROB_TILE_4: float = 0.1  # probabilité qu'une nouvelle tuile soit un 4 (sinon 2)

# Directions
UP: int = 0
DOWN: int = 1
LEFT: int = 2
RIGHT: int = 3


class Board:
    """Représente une partie de 2048 et toute sa mécanique de jeu.

    Attributs publics :
        grid (np.ndarray): grille 4x4, dtype int32, contenant les valeurs
            des tuiles (0 = case vide).
        score (int): score cumulé de la partie en cours.
    """

    def __init__(self) -> None:
        """Initialise une grille 4x4 vide puis y place deux tuiles de départ."""
        self.grid: np.ndarray = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.score: int = 0
        self.reset()

    # --- API publique ------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Remet le jeu à zéro et retourne la grille initiale.

        Returns:
            np.ndarray: la grille 4x4 contenant deux tuiles de départ.
        """
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        self.score = 0
        self._spawn_tile()
        self._spawn_tile()
        return self.grid

    def move(self, direction: int) -> tuple[int, bool]:
        """Effectue un mouvement dans la direction donnée.

        Si la grille change suite au mouvement, une nouvelle tuile est
        ajoutée aléatoirement (90 % de chance d'un 2, 10 % d'un 4).

        Args:
            direction (int): 0=haut, 1=bas, 2=gauche, 3=droite.

        Returns:
            tuple[int, bool]: (score gagné pendant ce mouvement,
                grille modifiée ou non).

        Raises:
            ValueError: si la direction n'est pas dans {0, 1, 2, 3}.
        """
        if direction not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError(f"Direction invalide : {direction!r} (attendu 0, 1, 2 ou 3)")

        # On ramène chaque mouvement à un glissement "vers la gauche" en
        # appliquant la bonne rotation, puis on remet la grille dans le bon sens.
        rotated = self._rotate_for_move(self.grid, direction)
        new_rotated, gained = self._slide_left(rotated)
        new_grid = self._rotate_back(new_rotated, direction)

        changed = not np.array_equal(new_grid, self.grid)
        if changed:
            self.grid = new_grid
            self.score += gained
            self._spawn_tile()

        return gained, changed

    def is_game_over(self) -> bool:
        """Indique si la partie est terminée (aucun mouvement possible).

        Returns:
            bool: True si plus aucune direction ne change la grille.
        """
        return len(self.get_valid_moves()) == 0

    def get_valid_moves(self) -> list[int]:
        """Retourne la liste des directions qui modifient la grille.

        Le calcul se fait sans toucher à l'état réel du jeu (pas de tuile
        ajoutée, score inchangé) : on simule simplement chaque glissement.

        Returns:
            list[int]: directions valides parmi {0, 1, 2, 3}.
        """
        valid: list[int] = []
        for direction in (UP, DOWN, LEFT, RIGHT):
            rotated = self._rotate_for_move(self.grid, direction)
            new_rotated, _ = self._slide_left(rotated)
            new_grid = self._rotate_back(new_rotated, direction)
            if not np.array_equal(new_grid, self.grid):
                valid.append(direction)
        return valid

    def get_max_tile(self) -> int:
        """Retourne la valeur de la plus grande tuile présente sur la grille.

        Returns:
            int: la valeur maximale (0 si la grille est vide).
        """
        return int(self.grid.max())

    # --- Mécanique interne -------------------------------------------------

    def _spawn_tile(self) -> None:
        """Ajoute une tuile (2 ou 4) sur une case vide choisie au hasard.

        Ne fait rien s'il n'y a aucune case vide.
        """
        empty_cells = np.argwhere(self.grid == 0)
        if empty_cells.size == 0:
            return
        idx = np.random.randint(len(empty_cells))
        row, col = empty_cells[idx]
        value = NEW_TILE_VALUES[1] if np.random.random() < PROB_TILE_4 else NEW_TILE_VALUES[0]
        self.grid[row, col] = value

    @staticmethod
    def _slide_left(grid: np.ndarray) -> tuple[np.ndarray, int]:
        """Glisse et fusionne toutes les lignes vers la gauche.

        C'est l'unique implémentation de la mécanique de fusion : les autres
        directions s'y ramènent par rotation. Deux tuiles identiques
        adjacentes fusionnent une seule fois par mouvement.

        Args:
            grid (np.ndarray): grille 4x4 à faire glisser vers la gauche.

        Returns:
            tuple[np.ndarray, int]: (nouvelle grille, score gagné).
        """
        new_grid = np.zeros_like(grid)
        gained = 0
        for r in range(grid.shape[0]):
            new_row, row_gain = Board._slide_row_left(grid[r])
            new_grid[r] = new_row
            gained += row_gain
        return new_grid, gained

    @staticmethod
    def _slide_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:
        """Glisse et fusionne une seule ligne vers la gauche.

        Args:
            row (np.ndarray): ligne 1D de longueur GRID_SIZE.

        Returns:
            tuple[np.ndarray, int]: (nouvelle ligne, score gagné sur la ligne).
        """
        # 1) On retire les zéros : on ne garde que les tuiles non vides.
        tiles = [int(v) for v in row if v != 0]

        # 2) On fusionne les paires identiques adjacentes, une seule fois.
        merged: list[int] = []
        gained = 0
        i = 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                fused = tiles[i] * 2
                merged.append(fused)
                gained += fused
                i += 2  # on saute la tuile fusionnée
            else:
                merged.append(tiles[i])
                i += 1

        # 3) On complète à droite avec des zéros pour garder la bonne taille.
        merged.extend([0] * (len(row) - len(merged)))
        return np.array(merged, dtype=row.dtype), gained

    @staticmethod
    def _rotate_for_move(grid: np.ndarray, direction: int) -> np.ndarray:
        """Tourne la grille pour que `direction` devienne un glissement gauche.

        Args:
            grid (np.ndarray): grille à transformer.
            direction (int): direction du mouvement réel.

        Returns:
            np.ndarray: grille orientée pour un glissement vers la gauche.
        """
        if direction == LEFT:
            return grid.copy()
        if direction == RIGHT:
            # Miroir horizontal : glisser à droite == miroir + glisser à gauche.
            return np.fliplr(grid)
        if direction == UP:
            # Les colonnes deviennent des lignes.
            return grid.T.copy()
        if direction == DOWN:
            return np.fliplr(grid.T)
        raise ValueError(f"Direction invalide : {direction!r}")

    @staticmethod
    def _rotate_back(grid: np.ndarray, direction: int) -> np.ndarray:
        """Annule la transformation de `_rotate_for_move` (opération inverse).

        Args:
            grid (np.ndarray): grille après glissement vers la gauche.
            direction (int): direction du mouvement réel.

        Returns:
            np.ndarray: grille remise dans l'orientation d'origine.
        """
        if direction == LEFT:
            return grid
        if direction == RIGHT:
            return np.fliplr(grid)
        if direction == UP:
            return grid.T
        if direction == DOWN:
            # Inverse de fliplr puis transpose : transpose(fliplr(...)).
            return np.fliplr(grid).T
        raise ValueError(f"Direction invalide : {direction!r}")
