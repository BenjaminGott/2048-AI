"""Agent PPO (Proximal Policy Optimization) pour le jeu 2048.

Ce module enveloppe `stable_baselines3.PPO` dans une petite classe pratique
qui gère aussi le **versioning** des modèles : chaque entraînement est sauvé
dans `models/{version}/` avec le modèle, ses hyperparamètres et ses résultats
d'évaluation. Cela permet de suivre la progression de l'agent au fil des
versions et de comparer avec la baseline aléatoire.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from env.game2048_env import Game2048Env

# --- Hyperparamètres PPO de départ -----------------------------------------
# Chaque valeur est commentée avec son rôle (projet d'apprentissage).
PPO_HYPERPARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",  # réseau fully-connected, adapté à l'entrée plate (16,)
    "learning_rate": 3e-4,  # pas d'apprentissage, valeur classique et stable pour PPO
    "n_steps": 2048,  # nb de pas collectés dans l'environnement avant chaque update
    "batch_size": 64,  # taille des mini-lots pour la descente de gradient
    "n_epochs": 10,  # nb de passes sur les données collectées à chaque update
    "gamma": 0.99,  # facteur d'actualisation : valorise les récompenses futures
    "gae_lambda": 0.95,  # lissage GAE : compromis biais/variance de l'avantage
    "clip_range": 0.2,  # clipping PPO : borne l'ampleur des mises à jour (stabilité)
    "ent_coef": 0.01,  # bonus d'entropie : pousse à explorer (évite de figer la politique)
    "verbose": 0,  # 0 = silencieux (on gère nous-mêmes l'affichage de progression)
}

# Statistiques de la baseline aléatoire (mesurées sur 100 parties) : sert de
# point de comparaison de référence dans les rapports de progression.
BASELINE_STATS: dict[str, float] = {
    "mean_score": 1050.0,
    "std_score": 470.0,
    "max_score": 3000,
    "mean_max_tile": 105.0,
    "max_tile_reached": 256,
    "pct_256": 5.0,
    "pct_512": 0.0,
    "pct_1024": 0.0,
}

# Seuils de tuiles dont on mesure le taux d'atteinte à l'évaluation.
TILE_THRESHOLDS: tuple[int, ...] = (256, 512, 1024)

# Garde-fous d'évaluation : une politique déterministe peut se retrouver à
# rejouer indéfiniment un coup invalide (la grille ne change pas). On borne
# donc le nombre de pas et le nombre de coups invalides consécutifs.
EVAL_MAX_STEPS: int = 10_000
EVAL_MAX_INVALID: int = 20


class PPOAgent:
    """Agent PPO entraînable, avec sauvegarde versionnée et évaluation."""

    def __init__(
        self,
        env: Game2048Env,
        model_dir: str = "models/",
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """Crée l'agent et instancie le modèle PPO.

        Args:
            env (Game2048Env): environnement d'entraînement.
            model_dir (str): dossier racine où sont sauvegardés les modèles.
            hyperparams (dict | None): surcharges optionnelles des hyperparamètres
                par défaut (utile pour des tests rapides, ex: n_steps réduit).
        """
        self.env = env
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Hyperparamètres effectifs = défauts + surcharges éventuelles.
        self.hyperparams: dict[str, Any] = {**PPO_HYPERPARAMS, **(hyperparams or {})}
        self.model = PPO(env=env, **self.hyperparams)

    # --- Entraînement / sauvegarde ----------------------------------------

    def train(self, total_timesteps: int, version: str, callback: Any | None = None) -> str:
        """Entraîne le modèle puis le sauvegarde sous `models/{version}/`.

        L'entraînement continue depuis l'état courant du modèle (warm start) :
        appeler `train` plusieurs fois enchaîne les versions sans repartir de zéro.

        Args:
            total_timesteps (int): nombre de pas d'entraînement pour cette version.
            version (str): nom de la version (ex: "v1").
            callback: callback SB3 optionnel (ex: affichage de progression).

        Returns:
            str: chemin du fichier modèle sauvegardé.
        """
        # reset_num_timesteps=False : on cumule le compteur de pas entre versions.
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        version_dir = os.path.join(self.model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        model_path = os.path.join(version_dir, "model.zip")
        self.model.save(model_path)

        meta = {
            "version": version,
            "total_timesteps": int(total_timesteps),
            "timesteps_cumulative": int(self.model.num_timesteps),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "hyperparams": self.hyperparams,
        }
        with open(os.path.join(version_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path

    def load(self, version: str) -> None:
        """Charge un modèle sauvegardé.

        Args:
            version (str): version à charger (ex: "v2").
        """
        model_path = os.path.join(self.model_dir, version, "model.zip")
        self.model = PPO.load(model_path, env=self.env)

    # --- Inférence / évaluation -------------------------------------------

    def predict(self, obs: np.ndarray) -> int:
        """Choisit la meilleure action selon la politique (sans exploration).

        Args:
            obs (np.ndarray): observation courante.

        Returns:
            int: action choisie (0=haut, 1=bas, 2=gauche, 3=droite).
        """
        action, _state = self.model.predict(obs, deterministic=True)
        return int(action)

    def evaluate(self, n_episodes: int = 50) -> dict[str, float]:
        """Joue `n_episodes` parties avec la politique courante et agrège les stats.

        Args:
            n_episodes (int): nombre de parties à jouer.

        Returns:
            dict: scores (moyen/écart-type/max), tuile max (moyenne/abs) et taux
                d'atteinte des tuiles 256/512/1024 (en %).
        """
        scores: list[int] = []
        max_tiles: list[int] = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            terminated = False
            steps = 0
            consecutive_invalid = 0

            while not terminated and steps < EVAL_MAX_STEPS:
                action = self.predict(obs)
                obs, _reward, terminated, _truncated, info = self.env.step(action)
                steps += 1
                # Détection d'un agent "bloqué" sur un coup invalide répété.
                if info["action_valid"]:
                    consecutive_invalid = 0
                else:
                    consecutive_invalid += 1
                    if consecutive_invalid >= EVAL_MAX_INVALID:
                        break

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
