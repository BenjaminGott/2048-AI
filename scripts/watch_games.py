"""Visualiseur de parties de 2048 jouées en simultané dans le terminal.

Affiche une grille de plusieurs parties jouées en parallèle par l'agent
aléatoire, mises à jour en temps réel (un coup par partie à chaque tick).
Pratique pour « voir » le comportement de l'agent et comparer plusieurs
parties d'un coup d'œil.

Avec l'option --record, chaque partie est enregistrée dans le dossier
`recordings/` et pourra être rejouée ensuite avec `scripts/replay_game.py`.

Lancement (depuis la racine du projet) :
    python scripts/watch_games.py                 # 6 parties
    python scripts/watch_games.py -n 12 -c 4      # 12 parties, 4 par ligne
    python scripts/watch_games.py -n 4 -d 0.2     # 4 parties, 0.2 s par tick
    python scripts/watch_games.py --record        # enregistre les parties
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from colorama import init as colorama_init

# Ajoute la racine du projet au chemin d'import (cf. scripts/evaluate_random.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.random_agent import RandomAgent
from env.game2048_env import Game2048Env
from game.render_text import board_block, grid_of_blocks
from game.replay import GameRecorder, parse_save_selection

CELL_WIDTH = 5  # cases compactes pour faire tenir plusieurs grilles côte à côte


@dataclass
class GameView:
    """État d'une partie suivie par le visualiseur.

    Attributs :
        game_id (int): numéro affiché de la partie.
        env (Game2048Env): environnement de cette partie.
        agent (RandomAgent): agent qui joue cette partie.
        obs (np.ndarray): dernière observation reçue.
        recorder (GameRecorder | None): enregistreur si l'option est active.
        done (bool): True si la partie est terminée.
        steps (int): nombre de coups joués.
        info (dict): dernières infos renvoyées par l'environnement.
    """

    game_id: int
    env: Game2048Env
    agent: RandomAgent
    obs: np.ndarray
    recorder: GameRecorder | None = None
    done: bool = False
    steps: int = 0
    info: dict = field(default_factory=dict)

    def play_one_step(self) -> None:
        """Joue un seul coup si la partie n'est pas terminée."""
        if self.done:
            return
        action = self.agent.select_action(self.obs)
        self.obs, _reward, terminated, _truncated, self.info = self.env.step(action)
        self.steps += 1
        self.done = terminated
        if self.recorder is not None:
            self.recorder.capture(self.env.board.grid, self.env.board.score, action)


def _build_block(game: GameView) -> list[str]:
    """Construit le bloc de texte (lignes) représentant une partie.

    Args:
        game (GameView): partie à afficher.

    Returns:
        list[str]: lignes du bloc (en-tête + grille encadrée).
    """
    status = "GAME OVER" if game.done else f"{game.steps} coups"
    score = game.info.get("score", 0)
    max_tile = game.info.get("max_tile", int(game.env.board.grid.max()))
    headers = [f"#{game.game_id}  score {score}", f"max {max_tile}  {status}"]
    return board_block(game.env.board.grid, header_lines=headers, cell_width=CELL_WIDTH)


def render_frame(games: list[GameView], cols: int) -> str:
    """Compose l'affichage complet : tous les blocs disposés en grille.

    Args:
        games (list[GameView]): parties à afficher.
        cols (int): nombre de parties par ligne.

    Returns:
        str: le cadre complet prêt à être imprimé.
    """
    return grid_of_blocks([_build_block(g) for g in games], cols=cols)


def make_games(n_games: int) -> list[GameView]:
    """Initialise `n_games` parties prêtes à être jouées.

    Args:
        n_games (int): nombre de parties à créer.

    Returns:
        list[GameView]: les parties initialisées (chacune avec son env/agent).
    """
    games: list[GameView] = []
    for i in range(1, n_games + 1):
        env = Game2048Env()
        agent = RandomAgent(env)
        obs, info = env.reset()  # pas de seed : chaque partie diffère naturellement
        # On enregistre toujours en mémoire : la sauvegarde sur disque sera
        # décidée à la fin (prompt). Capturer l'état initial (action=None).
        recorder = GameRecorder()
        recorder.capture(env.board.grid, env.board.score, action=None)
        games.append(
            GameView(game_id=i, env=env, agent=agent, obs=obs, recorder=recorder, info=info)
        )
    return games


def save_selected(games: list[GameView], indices: list[int]) -> str:
    """Enregistre sur disque les parties choisies, dans un sous-dossier daté.

    Args:
        games (list[GameView]): toutes les parties jouées.
        indices (list[int]): indices (0-based) des parties à sauvegarder.

    Returns:
        str: le dossier où les parties ont été écrites.
    """
    run_dir = os.path.join("recordings", time.strftime("run_%Y%m%d_%H%M%S"))
    for idx in indices:
        game = games[idx]
        assert game.recorder is not None
        game.recorder.save(
            directory=run_dir,
            name=f"game{game.game_id:02d}.json",
            extra_metadata={"agent": "random", "game_id": game.game_id},
        )
    return run_dir


def prompt_and_save(games: list[GameView], auto_all: bool) -> None:
    """Propose de sauvegarder les parties (toutes / aucune / une sélection).

    Args:
        games (list[GameView]): parties jouées.
        auto_all (bool): si True, sauvegarde tout sans poser de question
            (utile en mode non interactif, via --record).
    """
    if auto_all:
        indices = list(range(len(games)))
    elif sys.stdin.isatty():
        print("\nParties jouées :")
        for i, g in enumerate(games):
            print(
                f"  [{i}] score {g.info.get('score', 0):<6} "
                f"tuile max {g.info.get('max_tile', 0):<5} {g.steps} coups"
            )
        try:
            answer = input("\nSauvegarder lesquelles ? (all / none / ex: '0 2 3') : ")
        except EOFError:  # entrée indisponible (terminal non interactif)
            print("\n(entrée indisponible — aucune partie sauvegardée)")
            return
        indices = parse_save_selection(answer, len(games))
    else:
        # Pas de terminal interactif et pas de --record : on ne sauvegarde rien.
        print("\n(Astuce : --record pour sauvegarder automatiquement les parties.)")
        return

    if not indices:
        print("Aucune partie sauvegardée.")
        return

    run_dir = save_selected(games, indices)
    print(f"\n{len(indices)} partie(s) enregistrée(s) dans : {run_dir}")
    print(f"Pour les rejouer : python scripts/replay_game.py {run_dir}")


def watch(n_games: int, cols: int, delay: float, auto_save: bool) -> None:
    """Boucle principale : joue toutes les parties en parallèle et les affiche.

    Args:
        n_games (int): nombre de parties simultanées.
        cols (int): parties par ligne à l'écran.
        delay (float): pause (secondes) entre deux ticks.
        auto_save (bool): si True, sauvegarde toutes les parties sans prompt.
    """
    colorama_init()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # caractères de cadre en UTF-8

    games = make_games(n_games)

    while True:
        for game in games:
            game.play_one_step()

        # Repositionne le curseur en haut puis efface l'écran.
        print("\x1b[H\x1b[2J", end="")
        print(render_frame(games, cols))

        finished = sum(g.done for g in games)
        print(f"Parties terminées : {finished}/{n_games}  (Ctrl+C pour quitter)")

        if finished == n_games:
            break
        time.sleep(delay)

    best = max(games, key=lambda g: g.info.get("score", 0))
    print(
        f"\nToutes les parties terminées. "
        f"Meilleur score : {best.info.get('score', 0)} "
        f"(partie #{best.game_id}, tuile max {best.info.get('max_tile', 0)})."
    )

    prompt_and_save(games, auto_all=auto_save)


def main() -> None:
    """Point d'entrée : parse les arguments CLI puis lance le visualiseur."""
    parser = argparse.ArgumentParser(description="Visualise des parties de 2048 en simultané.")
    parser.add_argument("-n", "--games", type=int, default=6, help="nombre de parties (défaut 6)")
    parser.add_argument("-c", "--cols", type=int, default=3, help="parties par ligne (défaut 3)")
    parser.add_argument(
        "-d", "--delay", type=float, default=0.08, help="secondes entre deux ticks (défaut 0.08)"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="sauvegarde automatiquement toutes les parties (sans poser de question)",
    )
    args = parser.parse_args()

    try:
        watch(
            n_games=args.games,
            cols=max(1, args.cols),
            delay=max(0.0, args.delay),
            auto_save=args.record,
        )
    except KeyboardInterrupt:
        print("\nVisualisation interrompue.")


if __name__ == "__main__":
    main()
