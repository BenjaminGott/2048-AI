"""Environnement Gymnasium pour le jeu 2048.

Ce module enveloppe la logique pure de `game.board.Board` dans l'interface
standard `gymnasium.Env`, afin d'entraîner des agents de Reinforcement
Learning (PPO, MaskablePPO…) avec `stable-baselines3`.

Deux réglages permettent d'expérimenter (étape 4 — optimisation) :

* **obs_mode** :
  - `"flat"`  (défaut) : grille aplatie `(16,)`, encodée `log2(valeur)/17`.
    Simple, compatible avec une MlpPolicy.
  - `"onehot"` : tenseur `(C, 4, 4)` où chaque canal représente un exposant
    de tuile (canal 0 = case vide, canal k = tuile 2^k). Pensé pour un réseau
    convolutif (CNN) qui exploite la structure spatiale du plateau.

* **reward_mode** :
  - `"basic"` (défaut) : `log2(fusion) + 0.1·cases_vides`.
  - `"shaped"` : ajoute un shaping **potentiel** (Ng et al.) récompensant
    l'amélioration de la « qualité » du plateau (monotonie + grosse tuile dans
    un coin + cases vides). Cela guide l'agent vers des stratégies gagnantes.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.board import Board

# --- Constantes globales ---------------------------------------------------

GRID_SIZE: int = 4
GRID_CELLS: int = 16  # 4 x 4 cases aplaties
LOG2_MAX_TILE: float = 17.0  # 2^17 = 131072, tuile max théorique -> normalisation
N_ACTIONS: int = 4  # haut, bas, gauche, droite
NUM_TILE_CHANNELS: int = 16  # exposants 0..15 (canal 0 = vide), pour l'encodage one-hot

# Reward de base.
EMPTY_BONUS_WEIGHT: float = 0.1  # poids du bonus "cases vides"
INVALID_ACTION_PENALTY: float = -1.0  # pénalité d'un coup invalide

# Reward shaping (potentiel). Φ ∈ [0, 1] ; la récompense ajoutée est
# γ·Φ(s') − Φ(s), ce qui ne change pas la politique optimale (shaping potentiel).
SHAPING_GAMMA: float = 0.99
SHAPING_WEIGHT: float = 1.0
PHI_MONOTONICITY: float = 0.5  # poids de la monotonie dans le potentiel
PHI_CORNER: float = 0.3  # poids de "plus grosse tuile dans un coin"
PHI_EMPTY: float = 0.2  # poids des cases vides

# Directions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Game2048Env(gym.Env):
    """Environnement Gymnasium qui expose une partie de 2048 à un agent RL."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self, obs_mode: str = "flat", reward_mode: str = "basic") -> None:
        """Initialise l'environnement.

        Args:
            obs_mode (str): "flat" (vecteur (16,)) ou "onehot" (tenseur (C,4,4)).
            reward_mode (str): "basic" ou "shaped" (shaping potentiel).
        """
        super().__init__()
        if obs_mode not in ("flat", "onehot"):
            raise ValueError(f"obs_mode inconnu : {obs_mode!r}")
        if reward_mode not in ("basic", "shaped"):
            raise ValueError(f"reward_mode inconnu : {reward_mode!r}")

        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        self.board: Board = Board()
        self._prev_potential: float = 0.0

        if obs_mode == "flat":
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(GRID_CELLS,), dtype=np.float32
            )
        else:  # onehot
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(NUM_TILE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32
            )
        self.action_space = spaces.Discrete(N_ACTIONS)

    # --- API Gymnasium -----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Démarre une nouvelle partie."""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.board.reset()
        self._prev_potential = self._potential()
        return self._get_observation(), self._get_info(action_valid=True)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applique une action et fait avancer la partie d'un pas."""
        gained, changed = self.board.move(int(action))

        truncated = False
        terminated = self.board.is_game_over()
        observation = self._get_observation()

        # Coup invalide : pénalité, la partie continue, le potentiel ne bouge pas.
        if not changed:
            return observation, INVALID_ACTION_PENALTY, terminated, truncated, self._get_info(False)

        # 1) Reward de fusion (log2 du gain) — voir docstring du module.
        merge_reward = math.log2(gained) if gained > 0 else 0.0
        # 2) Bonus "cases vides".
        empty_cells = int(np.count_nonzero(self.board.grid == 0))
        reward = merge_reward + EMPTY_BONUS_WEIGHT * (empty_cells / GRID_CELLS)

        # 3) Shaping potentiel optionnel : on récompense l'amélioration de Φ.
        if self.reward_mode == "shaped":
            new_potential = self._potential()
            reward += SHAPING_WEIGHT * (SHAPING_GAMMA * new_potential - self._prev_potential)
            self._prev_potential = new_potential

        return observation, float(reward), terminated, truncated, self._get_info(True)

    # --- Action masking ----------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Renvoie le masque booléen (4,) des actions valides (pour MaskablePPO)."""
        valid = self.board.get_valid_moves()
        if not valid:
            return np.ones(N_ACTIONS, dtype=bool)
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[valid] = True
        return mask

    # --- Observation -------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Construit l'observation selon `obs_mode`."""
        if self.obs_mode == "flat":
            return self._obs_flat()
        return self._obs_onehot()

    def _obs_flat(self) -> np.ndarray:
        """Grille aplatie `(16,)`, encodée log2 normalisée dans [0, 1]."""
        flat = self.board.grid.astype(np.float32).flatten()
        obs = np.zeros_like(flat, dtype=np.float32)
        non_empty = flat > 0
        obs[non_empty] = np.log2(flat[non_empty]) / LOG2_MAX_TILE
        return obs

    def _obs_onehot(self) -> np.ndarray:
        """Tenseur one-hot `(C, 4, 4)` : un canal par exposant de tuile.

        Canal 0 = case vide ; canal k = tuile 2^k. Cet encodage donne au CNN
        une représentation « propre » (pas d'ordre numérique arbitraire entre
        les tuiles), ce qui facilite l'apprentissage des motifs spatiaux.
        """
        obs = np.zeros((NUM_TILE_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        grid = self.board.grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                value = int(grid[r, c])
                exponent = 0 if value == 0 else int(round(math.log2(value)))
                exponent = min(exponent, NUM_TILE_CHANNELS - 1)
                obs[exponent, r, c] = 1.0
        return obs

    # --- Reward shaping : potentiel du plateau -----------------------------

    def _potential(self) -> float:
        """Potentiel Φ(grille) ∈ [0, 1] : plus c'est élevé, mieux le plateau est rangé."""
        grid = self.board.grid
        empty = float(np.count_nonzero(grid == 0)) / GRID_CELLS
        return (
            PHI_MONOTONICITY * self._monotonicity(grid)
            + PHI_CORNER * self._max_in_corner(grid)
            + PHI_EMPTY * empty
        )

    @staticmethod
    def _monotonicity(grid: np.ndarray) -> float:
        """Fraction de paires adjacentes "ordonnées" sur lignes et colonnes ∈ [0.5, 1].

        Une grille bien rangée (valeurs croissantes ou décroissantes le long de
        chaque ligne/colonne) facilite les fusions ; on récompense donc cet ordre.
        """
        log_grid = np.where(grid > 0, np.log2(np.maximum(grid, 1)), 0.0)
        ordered = 0
        lines = list(log_grid) + list(log_grid.T)  # 4 lignes + 4 colonnes
        for line in lines:
            inc = sum(line[i] <= line[i + 1] for i in range(len(line) - 1))
            dec = sum(line[i] >= line[i + 1] for i in range(len(line) - 1))
            ordered += max(inc, dec)
        total = len(lines) * (GRID_SIZE - 1)  # 8 lignes * 3 paires = 24
        return ordered / total

    @staticmethod
    def _max_in_corner(grid: np.ndarray) -> float:
        """1.0 si la plus grande tuile est dans un coin, sinon 0.0."""
        max_val = grid.max()
        corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
        return 1.0 if max_val in corners else 0.0

    # --- Infos -------------------------------------------------------------

    def _get_info(self, action_valid: bool) -> dict[str, Any]:
        """Dictionnaire d'informations renvoyé par reset/step."""
        return {
            "score": int(self.board.score),
            "max_tile": int(self.board.get_max_tile()),
            "valid_moves": self.board.get_valid_moves(),
            "action_valid": action_valid,
        }
