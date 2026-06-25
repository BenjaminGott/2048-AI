"""Test de l'extracteur de features CNN (agents/features.py).

Ignoré si torch/stable-baselines3 ne sont pas installés.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

import torch as th  # noqa: E402

from agents.features import Grid2048CNN  # noqa: E402
from env.game2048_env import Game2048Env  # noqa: E402


def test_cnn_output_shape() -> None:
    """Le CNN transforme un batch (B, C, 4, 4) en (B, features_dim)."""
    env = Game2048Env(obs_mode="onehot")
    extractor = Grid2048CNN(env.observation_space, features_dim=64)

    batch = th.zeros(5, *env.observation_space.shape)
    out = extractor(batch)

    assert out.shape == (5, 64)
