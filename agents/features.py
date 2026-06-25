"""Extracteur de features convolutif pour le plateau 2048 (one-hot).

Le CNN par défaut de stable-baselines3 (`NatureCNN`) est conçu pour des images
de jeux Atari (grands noyaux 8×8…) et ne convient pas à une grille 4×4. On
définit donc un petit CNN adapté, branché via `policy_kwargs` sur la politique
de PPO/MaskablePPO.

Entrée attendue : observation one-hot de forme (C, 4, 4) produite par
`Game2048Env(obs_mode="onehot")`.
"""

from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class Grid2048CNN(BaseFeaturesExtractor):
    """Petit CNN qui transforme un plateau one-hot (C,4,4) en vecteur de features."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256) -> None:
        """Construit le réseau convolutif.

        Args:
            observation_space (spaces.Box): espace d'observation (C, 4, 4).
            features_dim (int): dimension du vecteur de features en sortie.
        """
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        # Noyaux 2×2 adaptés à une petite grille : ils captent les motifs locaux
        # (paires de tuiles voisines) sans réduire trop vite la résolution.
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        # On calcule dynamiquement la taille aplatie produite par le CNN.
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Passe avant : (B, C, 4, 4) -> (B, features_dim)."""
        return self.linear(self.cnn(observations))
