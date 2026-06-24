"""Visualiseur de parties de 2048 jouées en simultané dans le terminal.

Affiche une grille de plusieurs parties jouées en parallèle par l'agent
aléatoire, mises à jour en temps réel (un coup par partie à chaque tick).
Pratique pour « voir » le comportement de l'agent et comparer plusieurs
parties d'un coup d'œil.

Lancement (depuis la racine du projet) :
    python scripts/watch_games.py                 # 6 parties
    python scripts/watch_games.py -n 12 -c 4      # 12 parties, 4 par ligne
    python scripts/watch_games.py -n 4 -d 0.2     # 4 parties, 0.2 s par tick
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
from play import _MAX_COLOR_KEY, TILE_COLORS  # couleurs réutilisées

_RESET = "\x1b[0m"
CELL_WIDTH = 5  # largeur d'une case (plus compacte que play.py pour tenir plusieurs grilles)


@dataclass
class GameView:
    """État d'une partie suivie par le visualiseur.

    Attributs :
        game_id (int): numéro affiché de la partie.
        env (Game2048Env): environnement de cette partie.
        agent (RandomAgent): agent qui joue cette partie.
        obs (np.ndarray): dernière observation reçue.
        done (bool): True si la partie est terminée.
        steps (int): nombre de coups joués.
    """

    game_id: int
    env: Game2048Env
    agent: RandomAgent
    obs: np.ndarray
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


def _ansi_cell(value: int) -> str:
    """Représente une case colorée de largeur fixe (version compacte).

    Args:
        value (int): valeur de la tuile (0 = case vide).

    Returns:
        str: chaîne colorée (codes ANSI), de largeur visible CELL_WIDTH.
    """
    bg, fg = TILE_COLORS.get(min(value, _MAX_COLOR_KEY), TILE_COLORS[_MAX_COLOR_KEY])
    text = "" if value == 0 else str(value)
    return f"\x1b[48;5;{bg}m\x1b[38;5;{fg}m{text.rjust(CELL_WIDTH)}{_RESET}"


def _build_block(game: GameView) -> list[str]:
    """Construit le bloc de texte (lignes) représentant une partie.

    Toutes les lignes ont la même largeur visible, ce qui permet de poser
    plusieurs blocs côte à côte ensuite.

    Args:
        game (GameView): partie à afficher.

    Returns:
        list[str]: lignes du bloc (en-tête + grille encadrée).
    """
    grid = game.env.board.grid
    size = grid.shape[0]

    top = "┌" + "┬".join(["─" * CELL_WIDTH] * size) + "┐"
    mid = "├" + "┼".join(["─" * CELL_WIDTH] * size) + "┤"
    bot = "└" + "┴".join(["─" * CELL_WIDTH] * size) + "┘"
    width = len(top)  # largeur visible commune (top n'a pas de code ANSI)

    status = "GAME OVER" if game.done else f"{game.steps} coups"
    score = game.info.get("score", 0)
    max_tile = game.info.get("max_tile", int(grid.max()))

    header1 = f"#{game.game_id}  score {score}".ljust(width)
    header2 = f"max {max_tile}  {status}".ljust(width)

    lines: list[str] = [header1, header2, top]
    for r in range(size):
        cells = "│".join(_ansi_cell(int(v)) for v in grid[r])
        lines.append("│" + cells + "│")
        lines.append(mid if r < size - 1 else bot)
    return lines


def render_frame(games: list[GameView], cols: int) -> str:
    """Compose l'affichage complet : tous les blocs disposés en grille.

    Args:
        games (list[GameView]): parties à afficher.
        cols (int): nombre de parties par ligne.

    Returns:
        str: le cadre complet prêt à être imprimé.
    """
    out_lines: list[str] = []
    for start in range(0, len(games), cols):
        row_games = games[start : start + cols]
        blocks = [_build_block(g) for g in row_games]
        # Tous les blocs ont le même nombre de lignes : on les pose côte à côte.
        for line_idx in range(len(blocks[0])):
            out_lines.append("   ".join(block[line_idx] for block in blocks))
        out_lines.append("")  # ligne vide entre deux rangées de parties
    return "\n".join(out_lines)


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
        games.append(GameView(game_id=i, env=env, agent=agent, obs=obs, info=info))
    return games


def watch(n_games: int, cols: int, delay: float) -> None:
    """Boucle principale : joue toutes les parties en parallèle et les affiche.

    Args:
        n_games (int): nombre de parties simultanées.
        cols (int): parties par ligne à l'écran.
        delay (float): pause (secondes) entre deux ticks.
    """
    colorama_init()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # caractères de cadre en UTF-8

    games = make_games(n_games)

    while True:
        # Un coup par partie active.
        for game in games:
            game.play_one_step()

        # Réaffichage : on repositionne le curseur en haut (sans tout effacer
        # à chaque fois pour limiter le scintillement).
        print("\x1b[H\x1b[2J", end="")
        print(render_frame(games, cols))

        finished = sum(g.done for g in games)
        print(f"Parties terminées : {finished}/{n_games}  (Ctrl+C pour quitter)")

        if finished == n_games:
            break
        time.sleep(delay)

    # Bilan final.
    best = max(games, key=lambda g: g.info.get("score", 0))
    print(
        f"\nToutes les parties terminées. "
        f"Meilleur score : {best.info.get('score', 0)} "
        f"(partie #{best.game_id}, tuile max {best.info.get('max_tile', 0)})."
    )


def main() -> None:
    """Point d'entrée : parse les arguments CLI puis lance le visualiseur."""
    parser = argparse.ArgumentParser(description="Visualise des parties de 2048 en simultané.")
    parser.add_argument("-n", "--games", type=int, default=6, help="nombre de parties (défaut 6)")
    parser.add_argument("-c", "--cols", type=int, default=3, help="parties par ligne (défaut 3)")
    parser.add_argument(
        "-d", "--delay", type=float, default=0.08, help="secondes entre deux ticks (défaut 0.08)"
    )
    args = parser.parse_args()

    try:
        watch(n_games=args.games, cols=max(1, args.cols), delay=max(0.0, args.delay))
    except KeyboardInterrupt:
        print("\nVisualisation interrompue.")


if __name__ == "__main__":
    main()
