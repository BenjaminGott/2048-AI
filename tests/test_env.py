"""Tests unitaires de l'environnement Gymnasium (env/game2048_env.py).

Lancer les tests (depuis la racine du projet) :
    pytest
"""

from __future__ import annotations

import os
import sys

import numpy as np
from gymnasium.utils.env_checker import check_env

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.game2048_env import Game2048Env


def test_reset_returns_valid_observation() -> None:
    """reset() renvoie une observation de shape (16,) en float32."""
    env = Game2048Env()
    obs, info = env.reset(seed=0)

    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert "score" in info and "max_tile" in info and "valid_moves" in info


def test_observation_is_normalized() -> None:
    """L'observation reste bornée dans [0, 1]."""
    env = Game2048Env()
    obs, _ = env.reset(seed=42)

    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


def test_step_returns_correct_types() -> None:
    """step() renvoie (obs, reward, terminated, truncated, info) bien typés."""
    env = Game2048Env()
    env.reset(seed=0)
    # On choisit une action valide pour tester le cas nominal.
    valid_action = env.board.get_valid_moves()[0]
    obs, reward, terminated, truncated, info = env.step(valid_action)

    assert obs.shape == (16,)
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert truncated is False  # jamais de troncature dans cet env
    assert isinstance(info, dict)
    assert info["action_valid"] is True


def test_invalid_action_is_penalized() -> None:
    """Une action invalide renvoie un reward de -1.0 sans terminer la partie."""
    env = Game2048Env()
    env.reset(seed=0)

    # On force une grille où seul le mouvement "gauche" change la grille :
    # toute autre direction est alors invalide.
    env.board.grid = np.array(
        [
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [8, 0, 0, 0],
            [16, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    valid = env.board.get_valid_moves()
    invalid_action = next(a for a in range(4) if a not in valid)

    obs, reward, terminated, truncated, info = env.step(invalid_action)

    assert reward == -1.0
    assert terminated is False
    assert info["action_valid"] is False


def test_full_episode_terminates() -> None:
    """Une partie complète se termine sans lever d'exception."""
    env = Game2048Env()
    obs, _ = env.reset(seed=123)

    terminated = False
    steps = 0
    while not terminated:
        valid = env.board.get_valid_moves()
        action = valid[0] if valid else 0
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 100_000  # garde-fou contre une boucle infinie

    assert env.board.is_game_over()


def test_passes_gymnasium_env_checker() -> None:
    """L'environnement respecte l'API gymnasium (check_env officiel)."""
    env = Game2048Env()
    # check_env lève une exception si l'API n'est pas respectée.
    check_env(env, skip_render_check=True)
