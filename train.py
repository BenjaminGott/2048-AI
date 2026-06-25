"""Script d'entraînement de l'agent PPO avec versioning automatique.

Entraîne successivement plusieurs versions de l'agent (warm start), sauvegarde
chaque version dans `models/vN/`, l'évalue, et agrège la progression dans
`models/progress.json`.

Exemples (depuis la racine du projet) :
    python train.py                                  # 5 versions de 100k steps
    python train.py --timesteps 50000 --versions 3   # 3 versions de 50k steps
    python train.py --resume                         # reprend après la dernière version
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from agents.maskable_ppo_agent import MaskablePPOAgent
from agents.ppo_agent import BASELINE_STATS, PPOAgent
from env.game2048_env import Game2048Env

PROGRESS_FILE = "progress.json"


# --- Affichage de la progression en temps réel -----------------------------


class ProgressCallback(BaseCallback):
    """Affiche périodiquement l'avancement de l'entraînement dans le terminal."""

    def __init__(self, total_timesteps: int, version: str, print_every: int = 2048) -> None:
        """Configure l'affichage.

        Args:
            total_timesteps (int): objectif de pas pour cette version.
            version (str): nom de la version en cours.
            print_every (int): fréquence d'affichage (en pas).
        """
        super().__init__(verbose=0)
        self.total = total_timesteps
        self.version = version
        self.print_every = print_every
        self._start = time.time()
        self._next_threshold = print_every
        self._start_ts = 0

    def _on_training_start(self) -> None:
        self._start = time.time()
        # On mesure les pas via le compteur du modèle : robuste avec des
        # environnements vectorisés (où chaque _on_step vaut n_envs pas).
        self._start_ts = self.model.num_timesteps
        self._next_threshold = self.print_every

    def _on_step(self) -> bool:
        done = self.model.num_timesteps - self._start_ts
        if done >= self._next_threshold:
            self._next_threshold += self.print_every
            self._print_progress()
        return True

    def _print_progress(self) -> None:
        done = min(self.model.num_timesteps - self._start_ts, self.total)
        pct = 100.0 * done / max(1, self.total)
        elapsed = time.time() - self._start
        fps = done / elapsed if elapsed > 0 else 0.0
        # Récompense moyenne récente, si disponible dans le buffer SB3.
        rewards = [ep["r"] for ep in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        bar_len = 24
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "·" * (bar_len - filled)
        print(
            f"\r  [{self.version}] {bar} {pct:5.1f}%  "
            f"{done}/{self.total} pas  |  reward moy {mean_r:6.2f}  |  {fps:5.0f} pas/s",
            end="",
            flush=True,
        )


# --- Gestion du versioning et de progress.json -----------------------------


def list_versions(model_dir: str) -> list[str]:
    """Liste les versions existantes (dossiers `vN`) triées par numéro."""
    if not os.path.isdir(model_dir):
        return []
    versions = [d for d in os.listdir(model_dir) if re.fullmatch(r"v\d+", d)]
    return sorted(versions, key=lambda v: int(v[1:]))


def next_version_number(model_dir: str) -> int:
    """Renvoie le prochain numéro de version libre (1 si aucune)."""
    versions = list_versions(model_dir)
    return int(versions[-1][1:]) + 1 if versions else 1


def progress_path(model_dir: str) -> str:
    """Chemin du fichier de progression agrégé."""
    return os.path.join(model_dir, PROGRESS_FILE)


def load_progress(model_dir: str) -> dict[str, Any]:
    """Charge `progress.json` ou crée une structure vierge avec la baseline."""
    path = progress_path(model_dir)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"baseline": BASELINE_STATS, "versions": []}


def update_progress(
    model_dir: str,
    version: str,
    timesteps_total: int,
    eval_stats: dict[str, float],
    algo: str = "PPO",
) -> dict[str, Any]:
    """Ajoute (ou remplace) l'entrée d'une version dans `progress.json`.

    Args:
        model_dir (str): dossier des modèles.
        version (str): version concernée (ex: "v2").
        timesteps_total (int): total cumulé de pas d'entraînement.
        eval_stats (dict): résultats d'évaluation de la version.
        algo (str): algorithme utilisé (ex: "PPO", "MaskablePPO").

    Returns:
        dict: la progression mise à jour.
    """
    progress = load_progress(model_dir)
    entry = {
        "version": version,
        "algo": algo,
        "timesteps_total": int(timesteps_total),
        **eval_stats,
    }
    # On retire une éventuelle entrée du même nom puis on ré-ajoute, et on trie.
    others = [v for v in progress["versions"] if v["version"] != version]
    progress["versions"] = sorted(others + [entry], key=lambda v: int(v["version"][1:]))

    with open(progress_path(model_dir), "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    return progress


# --- Affichage du résumé d'une version -------------------------------------


def print_version_summary(
    version: str, timesteps_total: int, stats: dict[str, float], model_path: str
) -> None:
    """Affiche un encadré récapitulatif d'une version entraînée."""
    line = "═" * 48
    print(f"\n{line}")
    print(f"Version {version} — {timesteps_total:,} steps (cumulés)".replace(",", " "))
    print(line)
    print(f"Score moyen    : {stats['mean_score']:8.0f}  (± {stats['std_score']:.0f})")
    print(f"Score max      : {stats['max_score']:8.0f}")
    print(f"Tuile max moy. : {stats['mean_max_tile']:8.0f}")
    print(f"Tuile max abs. : {stats['max_tile_reached']:8d}")
    print(f"% → 256        : {stats['pct_256']:7.1f}%")
    print(f"% → 512        : {stats['pct_512']:7.1f}%")
    print(f"Modèle sauvegardé : {model_path}")
    print(line)


# --- Programme principal ----------------------------------------------------


def main() -> None:
    """Point d'entrée : parse les arguments et lance l'entraînement versionné."""
    parser = argparse.ArgumentParser(description="Entraîne l'agent PPO sur le 2048.")
    parser.add_argument("--timesteps", type=int, default=100_000, help="pas par version")
    parser.add_argument("--versions", type=int, default=5, help="nombre de versions à entraîner")
    parser.add_argument("--eval-episodes", type=int, default=50, help="parties d'évaluation")
    parser.add_argument("--resume", action="store_true", help="reprend depuis la dernière version")
    parser.add_argument("--model-dir", default="models", help="dossier des modèles")
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="utilise le PPO simple (sans masquage) au lieu de MaskablePPO",
    )
    parser.add_argument(
        "--n-envs", type=int, default=8, help="environnements parallèles (MaskablePPO)"
    )
    parser.add_argument(
        "--obs-mode",
        choices=["flat", "onehot"],
        default="flat",
        help="encodage de l'observation (onehot active le CNN)",
    )
    parser.add_argument(
        "--reward",
        choices=["basic", "shaped"],
        default="basic",
        help="fonction de reward (shaped = shaping potentiel)",
    )
    args = parser.parse_args()

    # Par défaut : MaskablePPO (masquage + vec-envs + meilleur modèle gardé).
    # --no-mask rebascule sur le PPO simple, utile comme point de comparaison.
    if args.no_mask:
        agent: PPOAgent | MaskablePPOAgent = PPOAgent(
            Game2048Env(obs_mode=args.obs_mode, reward_mode=args.reward),
            model_dir=args.model_dir,
        )
        algo = "PPO"
    else:
        agent = MaskablePPOAgent(
            model_dir=args.model_dir,
            n_envs=args.n_envs,
            obs_mode=args.obs_mode,
            reward_mode=args.reward,
        )
        algo = MaskablePPOAgent.algo_name
    print(f"Algorithme : {algo}  |  obs={args.obs_mode}  |  reward={args.reward}")

    start_number = 1
    if args.resume:
        existing = list_versions(args.model_dir)
        if existing:
            last = existing[-1]
            print(f"Reprise depuis {last}…")
            agent.load(last)
            start_number = int(last[1:]) + 1
        else:
            print("Aucune version existante : démarrage à v1.")

    for i in range(args.versions):
        version = f"v{start_number + i}"
        print(f"\n>>> Entraînement de {version} ({args.timesteps:,} pas)…".replace(",", " "))

        callback = ProgressCallback(args.timesteps, version)
        model_path = agent.train(args.timesteps, version, callback=callback)
        print()  # newline après la barre de progression

        stats = agent.evaluate(n_episodes=args.eval_episodes)
        version_dir = os.path.join(args.model_dir, version)
        with open(os.path.join(version_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        timesteps_total = int(agent.model.num_timesteps)
        update_progress(args.model_dir, version, timesteps_total, stats, algo=algo)
        print_version_summary(version, timesteps_total, stats, model_path)

    print("\nEntraînement terminé. Compare les versions avec : python evaluate.py")


if __name__ == "__main__":
    main()
