"""Agent baseline aléatoire pour le jeu 2048.

Cet agent ne contient aucun apprentissage : il joue des coups au hasard
(parmi les coups valides). Il sert de **référence** (baseline) : tout agent
de Reinforcement Learning entraîné devra faire significativement mieux que
ces chiffres pour prouver qu'il a réellement appris quelque chose.
"""

from __future__ import annotations

import numpy as np

from env.game2048_env import Game2048Env


class RandomAgent:
    """Agent qui choisit ses actions au hasard parmi les coups valides."""

    def __init__(self, env: Game2048Env) -> None:
        """Mémorise l'environnement sur lequel l'agent joue.

        Args:
            env (Game2048Env): environnement 2048 à jouer.
        """
        self.env = env

    def select_action(self, obs: np.ndarray) -> int:
        """Choisit une action aléatoire parmi les mouvements valides.

        On évite volontairement de jouer un coup invalide (qui ne change pas
        la grille et serait pénalisé) : c'est ce qui rend cette baseline un
        peu plus forte qu'un pur tirage uniforme sur les 4 actions.

        Args:
            obs (np.ndarray): observation courante (non utilisée par cet agent
                aléatoire, mais présente pour respecter l'interface d'un agent).

        Returns:
            int: l'action choisie (0=haut, 1=bas, 2=gauche, 3=droite).
        """
        valid_moves = self.env.board.get_valid_moves()
        if valid_moves:
            return int(np.random.choice(valid_moves))
        # Aucun coup valide (partie terminée) : on renvoie une action quelconque.
        return int(np.random.randint(self.env.action_space.n))

    def play_episode(self) -> dict[str, int]:
        """Joue une partie complète jusqu'au game over.

        Returns:
            dict: statistiques de la partie :
                {"score": int, "max_tile": int, "steps": int}.
        """
        obs, _info = self.env.reset()
        terminated = False
        steps = 0

        while not terminated:
            action = self.select_action(obs)
            obs, _reward, terminated, _truncated, info = self.env.step(action)
            steps += 1

        return {
            "score": int(info["score"]),
            "max_tile": int(info["max_tile"]),
            "steps": steps,
        }
