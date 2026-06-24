"""Chargement d'un agent entraîné selon les métadonnées de sa version.

Une version sauvegardée contient un `meta.json` qui indique l'algorithme
utilisé (`PPO` ou `MaskablePPO`). Ce module lit cette information et instancie
le bon type d'agent, pour que l'évaluation et la visualisation n'aient pas à
s'en soucier.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from agents.ppo_agent import PPOAgent
from env.game2048_env import Game2048Env

if TYPE_CHECKING:  # uniquement pour le typage (évite d'importer sb3-contrib si inutile)
    from agents.maskable_ppo_agent import MaskablePPOAgent


def read_algo(version: str, model_dir: str) -> str:
    """Lit l'algorithme d'une version depuis son `meta.json` (défaut "PPO")."""
    meta_path = os.path.join(model_dir, version, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("algo", "PPO")
    return "PPO"


def make_agent_for(version: str, model_dir: str, n_envs: int = 1) -> PPOAgent | MaskablePPOAgent:
    """Instancie le bon type d'agent (non chargé) pour une version donnée.

    Args:
        version (str): version concernée (ex: "v3").
        model_dir (str): dossier des modèles.
        n_envs (int): nb d'environnements (utile seulement pour MaskablePPO).

    Returns:
        Un agent prêt à recevoir `.load(version)`.
    """
    if read_algo(version, model_dir) == "MaskablePPO":
        # Import local : sb3-contrib n'est nécessaire que pour les modèles masqués.
        from agents.maskable_ppo_agent import MaskablePPOAgent

        return MaskablePPOAgent(Game2048Env(), model_dir=model_dir, n_envs=n_envs)
    return PPOAgent(Game2048Env(), model_dir=model_dir)
