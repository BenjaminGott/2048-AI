"""Environnement Gymnasium pour le jeu 2048.

Ce module enveloppe la logique pure de `game.board.Board` dans l'interface
standard `gymnasium.Env`, afin de pouvoir entraîner des agents de
Reinforcement Learning (PPO, DQN…) avec des bibliothèques comme
`stable-baselines3`.

Choix de design importants (projet d'apprentissage — on explicite le POURQUOI) :

* **Encodage log2 de la grille** : les valeurs des tuiles croissent de façon
  exponentielle (2, 4, 8, …, 2048…). Un réseau de neurones apprend bien plus
  facilement sur une échelle linéaire : on prend donc le log2 de chaque tuile
  (2→1, 4→2, 8→3, …) plutôt que la valeur brute.
* **Normalisation entre 0 et 1** : on divise par 17 car 2^17 = 131072 est la
  plus grande tuile théoriquement atteignable sur une grille 4×4. Garder les
  entrées dans [0, 1] stabilise et accélère l'apprentissage.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.board import Board

# --- Constantes globales ---------------------------------------------------

GRID_CELLS: int = 16  # 4 x 4 cases aplaties
LOG2_MAX_TILE: float = 17.0  # 2^17 = 131072, tuile max théorique -> facteur de normalisation
N_ACTIONS: int = 4  # haut, bas, gauche, droite

# Pondération du bonus "cases vides" dans le reward (voir step()).
EMPTY_BONUS_WEIGHT: float = 0.1
# Pénalité appliquée à une action qui ne change pas la grille (coup invalide).
INVALID_ACTION_PENALTY: float = -1.0


class Game2048Env(gym.Env):
    """Environnement Gymnasium qui expose une partie de 2048 à un agent RL.

    Attributs :
        board (Board): instance de la logique de jeu (état réel de la partie).
        observation_space (spaces.Box): vecteur (16,) float32 dans [0, 1].
        action_space (spaces.Discrete): 4 actions (0=haut, 1=bas, 2=gauche, 3=droite).
    """

    # `metadata` est attendu par gymnasium ; pas de rendu graphique ici.
    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self) -> None:
        """Initialise l'environnement, ses espaces d'observation et d'action."""
        super().__init__()
        self.board: Board = Board()

        # Observation : la grille 4x4 aplatie, encodée log2 et normalisée dans [0, 1].
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(GRID_CELLS,), dtype=np.float32
        )
        # Action : 4 directions discrètes.
        self.action_space = spaces.Discrete(N_ACTIONS)

    # --- API Gymnasium -----------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Démarre une nouvelle partie.

        Args:
            seed (int | None): graine pour la reproductibilité.
            options (dict | None): non utilisé ici (présent pour respecter l'API).

        Returns:
            tuple[np.ndarray, dict]: (observation initiale, dictionnaire d'infos).
        """
        # Initialise le générateur aléatoire de gymnasium (self.np_random).
        super().reset(seed=seed)
        # Board s'appuie sur np.random : on le seed aussi pour des parties
        # reproductibles quand une graine est fournie.
        if seed is not None:
            np.random.seed(seed)

        self.board.reset()
        observation = self._get_observation()
        return observation, self._get_info(action_valid=True)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applique une action et fait avancer la partie d'un pas.

        Args:
            action (int): direction jouée (0=haut, 1=bas, 2=gauche, 3=droite).

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
                - terminated : True si la partie est finie (plus aucun coup possible).
                - truncated : toujours False (pas de limite de durée artificielle).
        """
        # On compte les cases vides AVANT de jouer n'est pas nécessaire :
        # le bonus se calcule sur l'état résultant pour refléter l'espace restant.
        gained, changed = self.board.move(int(action))

        # gymnasium n'introduit pas de coupure temporelle ici.
        truncated = False
        terminated = self.board.is_game_over()
        observation = self._get_observation()

        # --- Cas 1 : action invalide (la grille n'a pas bougé) -------------
        # On pénalise pour décourager l'agent de "gâcher" des coups, mais la
        # partie continue (on ne termine pas sur un simple coup invalide).
        if not changed:
            info = self._get_info(action_valid=False)
            return observation, INVALID_ACTION_PENALTY, terminated, truncated, info

        # --- Cas 2 : action valide -> reward en deux composantes -----------

        # 1) Reward de fusion : log2 du gain de score du tour.
        #    Pourquoi log2 ? Fusionner deux 1024 (gain 2048) doit valoir plus
        #    que fusionner deux 2 (gain 4), mais pas 512x plus : l'échelle log
        #    garde des magnitudes de reward raisonnables et comparables.
        merge_reward = math.log2(gained) if gained > 0 else 0.0

        # 2) Bonus "cases vides" : plus il reste de place, mieux c'est.
        #    Pourquoi ? Une grille pleine mène rapidement au game over ;
        #    on encourage donc l'agent à fusionner et garder de l'espace.
        empty_cells = int(np.count_nonzero(self.board.grid == 0))
        empty_bonus = empty_cells / GRID_CELLS

        reward = merge_reward + EMPTY_BONUS_WEIGHT * empty_bonus

        info = self._get_info(action_valid=True)
        return observation, float(reward), terminated, truncated, info

    # --- Action masking ----------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Renvoie le masque des actions valides (pour MaskablePPO).

        Une action est « valide » si elle modifie la grille. Masquer les coups
        invalides évite à l'agent de les choisir : il n'apprend que sur des
        coups légaux, ce qui accélère et stabilise fortement l'apprentissage.

        Returns:
            np.ndarray: tableau booléen de forme (4,) — True = action autorisée.
                Si aucune action n'est valide (partie finie), tout est True pour
                éviter un masque entièrement faux (cas dégénéré non bloquant).
        """
        valid = self.board.get_valid_moves()
        mask = np.zeros(N_ACTIONS, dtype=bool)
        if not valid:
            return np.ones(N_ACTIONS, dtype=bool)
        mask[valid] = True
        return mask

    # --- Utilitaires internes ----------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Convertit la grille du Board en observation log2 normalisée.

        Returns:
            np.ndarray: vecteur (16,) float32 dans [0, 1].
                Case vide -> 0.0 ; sinon log2(valeur) / 17.
        """
        flat = self.board.grid.astype(np.float32).flatten()
        obs = np.zeros_like(flat, dtype=np.float32)
        # On n'applique le log2 que sur les cases non vides (log2(0) est indéfini).
        non_empty = flat > 0
        obs[non_empty] = np.log2(flat[non_empty]) / LOG2_MAX_TILE
        return obs

    def _get_info(self, action_valid: bool) -> dict[str, Any]:
        """Construit le dictionnaire d'informations renvoyé par reset/step.

        Args:
            action_valid (bool): si la dernière action a modifié la grille.

        Returns:
            dict: score courant, meilleure tuile, coups valides, validité de l'action.
        """
        return {
            "score": int(self.board.score),
            "max_tile": int(self.board.get_max_tile()),
            "valid_moves": self.board.get_valid_moves(),
            "action_valid": action_valid,
        }
