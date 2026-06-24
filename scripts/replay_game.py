"""Relecture de parties de 2048 enregistrées.

Rejoue visuellement une ou plusieurs parties sauvegardées (dossier
`recordings/` par défaut), de façon synchronisée, dans le terminal.

Exemples (depuis la racine du projet) :
    python scripts/replay_game.py                       # liste les parties dispo
    python scripts/replay_game.py --list                # idem, explicitement
    python scripts/replay_game.py 0                      # rejoue la partie d'index 0
    python scripts/replay_game.py 0 2 3                  # rejoue ces 3 parties
    python scripts/replay_game.py all                    # rejoue toutes les parties
    python scripts/replay_game.py recordings/run_xxx     # rejoue tout un dossier
    python scripts/replay_game.py game01.json -d 0.2     # par nom de fichier, plus lent
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from colorama import init as colorama_init

# Ajoute la racine du projet au chemin d'import (cf. scripts/evaluate_random.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.render_text import board_block, grid_of_blocks
from game.replay import (
    DEFAULT_RECORDINGS_DIR,
    list_replays,
    load_replay,
    resolve_replay_paths,
)

CELL_WIDTH = 5  # cases compactes (plusieurs parties possibles côte à côte)


def print_listing(directory: str) -> None:
    """Affiche la liste des parties enregistrées disponibles.

    Args:
        directory (str): dossier à explorer.
    """
    replays = list_replays(directory)
    if not replays:
        print(f"Aucune partie enregistrée dans '{directory}'.")
        print("Génère-en avec : python scripts/watch_games.py --record")
        return

    print(f"Parties enregistrées dans '{directory}' :")
    for i, path in enumerate(replays):
        meta = load_replay(path).get("metadata", {})
        print(
            f"  [{i}] {os.path.basename(path):<24} "
            f"score={meta.get('score', '?'):<6} "
            f"max_tile={meta.get('max_tile', '?'):<5} "
            f"coups={meta.get('steps', '?')}"
        )
    print("\nRejoue avec : python scripts/replay_game.py <index|all|fichier>")


def _build_block(name: str, frame: dict, frame_idx: int, total: int) -> list[str]:
    """Construit le bloc d'affichage d'une partie à un instant donné.

    Args:
        name (str): nom de la partie (pour l'en-tête).
        frame (dict): frame courante {"grid", "score", "action"}.
        frame_idx (int): index de la frame affichée.
        total (int): nombre total de frames de cette partie.

    Returns:
        list[str]: lignes du bloc.
    """
    at_end = frame_idx >= total - 1
    status = "FIN" if at_end else f"{frame_idx}/{total - 1}"
    headers = [f"{name}", f"score {frame.get('score', 0)}  {status}"]
    return board_block(frame["grid"], header_lines=headers, cell_width=CELL_WIDTH)


def replay(paths: list[str], cols: int, delay: float) -> None:
    """Rejoue les parties données de façon synchronisée.

    Toutes les parties avancent d'une frame par tick. Les parties plus courtes
    « gèlent » sur leur dernière image jusqu'à ce que la plus longue se termine.

    Args:
        paths (list[str]): fichiers de parties à rejouer.
        cols (int): parties par ligne à l'écran.
        delay (float): pause (secondes) entre deux frames.
    """
    colorama_init()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    games = [(os.path.basename(p), load_replay(p)["frames"]) for p in paths]
    max_len = max(len(frames) for _, frames in games)

    for t in range(max_len):
        blocks = []
        for name, frames in games:
            idx = min(t, len(frames) - 1)
            blocks.append(_build_block(name, frames[idx], idx, len(frames)))

        print("\x1b[H\x1b[2J", end="")
        print(grid_of_blocks(blocks, cols=cols))
        print(f"Frame {t}/{max_len - 1}  (Ctrl+C pour quitter)")
        time.sleep(delay)

    print("\nRelecture terminée.")


def main() -> None:
    """Point d'entrée : parse les arguments CLI puis lance la relecture."""
    parser = argparse.ArgumentParser(description="Rejoue des parties de 2048 enregistrées.")
    parser.add_argument(
        "targets",
        nargs="*",
        help="index(es), nom(s) de fichier, dossier(s) ou 'all' (vide = liste)",
    )
    parser.add_argument("--dir", default=DEFAULT_RECORDINGS_DIR, help="dossier des enregistrements")
    parser.add_argument("-c", "--cols", type=int, default=3, help="parties par ligne (défaut 3)")
    parser.add_argument(
        "-d", "--delay", type=float, default=0.12, help="secondes entre deux frames (défaut 0.12)"
    )
    parser.add_argument("-l", "--list", action="store_true", help="liste les parties disponibles")
    args = parser.parse_args()

    if args.list or not args.targets:
        print_listing(args.dir)
        return

    paths = resolve_replay_paths(args.targets, args.dir)
    if not paths:
        print("Aucune partie à rejouer.")
        return

    try:
        replay(paths, cols=max(1, args.cols), delay=max(0.0, args.delay))
    except KeyboardInterrupt:
        print("\nRelecture interrompue.")


if __name__ == "__main__":
    main()
