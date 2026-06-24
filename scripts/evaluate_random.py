"""Évaluation de l'agent aléatoire baseline sur N parties de 2048.

Ce script fait jouer l'agent aléatoire un grand nombre de fois et agrège les
résultats (score moyen, meilleure tuile, taux d'atteinte de certaines tuiles).
Ces chiffres constituent la **baseline** à battre pour le futur agent RL.

Lancement (depuis la racine du projet) :
    python scripts/evaluate_random.py            # 100 parties
    python scripts/evaluate_random.py 500        # 500 parties
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ajoute la racine du projet au chemin d'import pour que `import env` /
# `import agents` fonctionnent même lancé via `python scripts/evaluate_random.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.random_agent import RandomAgent
from env.game2048_env import Game2048Env

# Seuils de tuiles dont on veut connaître le taux d'atteinte.
TILE_THRESHOLDS: tuple[int, ...] = (256, 512, 1024, 2048)


def evaluate(n_episodes: int) -> dict[str, object]:
    """Fait jouer l'agent aléatoire `n_episodes` fois et agrège les stats.

    Args:
        n_episodes (int): nombre de parties à jouer.

    Returns:
        dict: statistiques agrégées (scores, tuiles, taux d'atteinte).
    """
    env = Game2048Env()
    agent = RandomAgent(env)

    scores: list[int] = []
    max_tiles: list[int] = []

    for i in range(1, n_episodes + 1):
        stats = agent.play_episode()
        scores.append(stats["score"])
        max_tiles.append(stats["max_tile"])
        # Barre de progression simple, réécrite sur la même ligne.
        print(f"\rEpisode {i}/{n_episodes}", end="", flush=True)

    print()  # saut de ligne après la progression

    scores_arr = np.array(scores)
    tiles_arr = np.array(max_tiles)

    return {
        "n_episodes": n_episodes,
        "score_mean": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "score_max": int(scores_arr.max()),
        "tile_mean": float(tiles_arr.mean()),
        "tile_max": int(tiles_arr.max()),
        "reach_rates": {t: float(np.mean(tiles_arr >= t) * 100.0) for t in TILE_THRESHOLDS},
    }


def print_report(results: dict[str, object]) -> None:
    """Affiche un rapport lisible des résultats de l'évaluation.

    Args:
        results (dict): sortie de `evaluate`.
    """
    n = results["n_episodes"]
    print(f"\n=== Résultats agent aléatoire ({n} épisodes) ===")
    print(f"Score moyen      : {results['score_mean']:8.1f}  (± {results['score_std']:7.1f})")
    print(f"Score max        : {results['score_max']:8d}")
    print(f"Tuile max moy.   : {results['tile_mean']:8.1f}")
    print(f"Tuile max abs.   : {results['tile_max']:8d}")
    for threshold, rate in results["reach_rates"].items():  # type: ignore[union-attr]
        print(f"% atteignant {threshold:<4} : {rate:7.1f}%")


def main() -> None:
    """Point d'entrée : lit le nombre d'épisodes en argument CLI puis évalue."""
    n_episodes = 100
    if len(sys.argv) > 1:
        try:
            n_episodes = int(sys.argv[1])
        except ValueError:
            print(f"Argument invalide : {sys.argv[1]!r} (entier attendu).")
            sys.exit(1)

    results = evaluate(n_episodes)
    print_report(results)


if __name__ == "__main__":
    main()
