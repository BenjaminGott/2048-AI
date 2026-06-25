"""Agent PPO avec masquage d'actions (MaskablePPO) pour le 2048.

Trois améliorations par rapport à `PPOAgent` pour que l'agent apprenne
réellement mieux :

1. **Action masking** : l'agent ne peut choisir que des coups valides (qui
   changent la grille), via `Game2048Env.action_masks()`. Il ne gaspille donc
   plus de pas sur des coups illégaux — c'est le gain le plus important au 2048.
2. **Environnements vectorisés** : plusieurs parties tournent en parallèle pour
   collecter les données bien plus vite (`n_envs`).
3. **Sauvegarde du meilleur modèle** : un `MaskableEvalCallback` évalue
   régulièrement et conserve le meilleur modèle rencontré (`best_model.zip`),
   pas seulement le dernier (PPO peut régresser temporairement).

L'API publique (train / load / predict / evaluate) est identique à `PPOAgent`
pour pouvoir comparer les deux approches.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.features import Grid2048CNN
from agents.ppo_agent import (
    EVAL_MAX_STEPS,
    PPO_HYPERPARAMS,
)
from env.game2048_env import Game2048Env


def _mask_fn(env: Game2048Env) -> np.ndarray:
    """Fonction de masque utilisée par le wrapper ActionMasker."""
    return env.action_masks()


def make_masked_env(obs_mode: str = "flat", reward_mode: str = "basic") -> ActionMasker:
    """Crée un `Game2048Env` (avec les modes voulus) exposant le masque d'actions."""
    return ActionMasker(Game2048Env(obs_mode=obs_mode, reward_mode=reward_mode), _mask_fn)


def _json_safe(value: Any) -> Any:
    """Rend un dict d'hyperparamètres sérialisable en JSON.

    Les valeurs non sérialisables (ex: la classe d'extracteur de features dans
    `policy_kwargs`) sont remplacées par leur nom lisible.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return getattr(value, "__name__", str(value))


class MaskablePPOAgent:
    """Agent MaskablePPO entraînable, avec vec-envs et sauvegarde du meilleur."""

    algo_name = "MaskablePPO"

    def __init__(
        self,
        env: Game2048Env | None = None,
        model_dir: str = "models/",
        n_envs: int = 8,
        hyperparams: dict[str, Any] | None = None,
        obs_mode: str = "flat",
        reward_mode: str = "basic",
    ) -> None:
        """Crée l'agent et instancie le modèle MaskablePPO.

        Args:
            env (Game2048Env | None): environnement d'évaluation/prédiction. Si
                None, un environnement masqué est créé avec les modes choisis.
            model_dir (str): dossier racine de sauvegarde des modèles.
            n_envs (int): nombre d'environnements parallèles pour l'entraînement.
            hyperparams (dict | None): surcharges des hyperparamètres par défaut.
            obs_mode (str): "flat" ou "onehot" (voir Game2048Env).
            reward_mode (str): "basic" ou "shaped" (voir Game2048Env).
        """
        self.model_dir = model_dir
        self.n_envs = max(1, n_envs)
        self.obs_mode = obs_mode
        self.reward_mode = reward_mode
        os.makedirs(model_dir, exist_ok=True)

        # Fabrique d'environnements partageant les mêmes modes (réutilisée pour
        # l'env d'éval du callback, sinon ses observations ne colleraient pas).
        self._env_factory = partial(make_masked_env, obs_mode=obs_mode, reward_mode=reward_mode)

        # Environnement d'évaluation/prédiction (masqué, un seul).
        self.env: ActionMasker = (
            ActionMasker(env, _mask_fn) if env is not None else self._env_factory()
        )

        # Environnements d'entraînement vectorisés (collecte parallèle).
        self.train_env = DummyVecEnv([self._env_factory for _ in range(self.n_envs)])

        self.hyperparams: dict[str, Any] = {**PPO_HYPERPARAMS, **(hyperparams or {})}
        # En mode one-hot, on branche le CNN dédié à la grille 4×4.
        if obs_mode == "onehot":
            policy_kwargs = dict(self.hyperparams.get("policy_kwargs", {}))
            policy_kwargs.setdefault("features_extractor_class", Grid2048CNN)
            policy_kwargs.setdefault("features_extractor_kwargs", {"features_dim": 256})
            self.hyperparams["policy_kwargs"] = policy_kwargs

        self.model = MaskablePPO(env=self.train_env, **self.hyperparams)

    # --- Entraînement / sauvegarde ----------------------------------------

    def train(self, total_timesteps: int, version: str, callback: Any | None = None) -> str:
        """Entraîne le modèle, sauve le dernier ET le meilleur modèle évalué.

        Args:
            total_timesteps (int): pas d'entraînement pour cette version.
            version (str): nom de la version (ex: "v1").
            callback: callback SB3 optionnel (ex: barre de progression).

        Returns:
            str: chemin du fichier modèle (le dernier, `model.zip`).
        """
        version_dir = os.path.join(self.model_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Callback qui évalue périodiquement et garde le meilleur modèle.
        # L'env d'éval doit utiliser les mêmes modes (obs/reward) que l'agent.
        eval_cb = MaskableEvalCallback(
            self._env_factory(),
            best_model_save_path=version_dir,
            n_eval_episodes=10,
            eval_freq=max(1, total_timesteps // 5 // self.n_envs),
            deterministic=True,
            verbose=0,
        )
        callbacks: list[BaseCallback] = [eval_cb]
        if callback is not None:
            callbacks.append(callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
            progress_bar=False,
        )

        model_path = os.path.join(version_dir, "model.zip")
        self.model.save(model_path)

        meta = {
            "version": version,
            "algo": self.algo_name,
            "n_envs": self.n_envs,
            "obs_mode": self.obs_mode,
            "reward_mode": self.reward_mode,
            "total_timesteps": int(total_timesteps),
            "timesteps_cumulative": int(self.model.num_timesteps),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hyperparams": _json_safe(self.hyperparams),
        }
        with open(os.path.join(version_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path

    def load(self, version: str, prefer_best: bool = True) -> None:
        """Charge un modèle sauvegardé.

        Args:
            version (str): version à charger (ex: "v2").
            prefer_best (bool): charge `best_model.zip` s'il existe, sinon `model.zip`.
        """
        version_dir = os.path.join(self.model_dir, version)
        best = os.path.join(version_dir, "best_model.zip")
        path = (
            best if prefer_best and os.path.isfile(best) else os.path.join(version_dir, "model.zip")
        )
        self.model = MaskablePPO.load(path, env=self.train_env)

    # --- Inférence / évaluation -------------------------------------------

    def predict(self, obs: np.ndarray) -> int:
        """Choisit la meilleure action légale selon la politique (sans exploration).

        Args:
            obs (np.ndarray): observation courante.

        Returns:
            int: action choisie (0=haut, 1=bas, 2=gauche, 3=droite).
        """
        mask = self.env.action_masks()
        action, _state = self.model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    def evaluate(self, n_episodes: int = 50) -> dict[str, float]:
        """Joue `n_episodes` parties avec la politique courante et agrège les stats.

        Args:
            n_episodes (int): nombre de parties à jouer.

        Returns:
            dict: mêmes clés que `PPOAgent.evaluate` (scores, tuiles, taux %).
        """
        scores: list[int] = []
        max_tiles: list[int] = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            terminated = False
            steps = 0
            while not terminated and steps < EVAL_MAX_STEPS:
                action = self.predict(obs)
                obs, _reward, terminated, _truncated, info = self.env.step(action)
                steps += 1
            scores.append(int(info["score"]))
            max_tiles.append(int(info["max_tile"]))

        scores_arr = np.array(scores)
        tiles_arr = np.array(max_tiles)
        return {
            "mean_score": float(scores_arr.mean()),
            "std_score": float(scores_arr.std()),
            "max_score": int(scores_arr.max()),
            "mean_max_tile": float(tiles_arr.mean()),
            "max_tile_reached": int(tiles_arr.max()),
            "pct_256": float(np.mean(tiles_arr >= 256) * 100.0),
            "pct_512": float(np.mean(tiles_arr >= 512) * 100.0),
            "pct_1024": float(np.mean(tiles_arr >= 1024) * 100.0),
        }
