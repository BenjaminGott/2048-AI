"""Relecture GRAPHIQUE (pygame) de parties de 2048 enregistrées.

Ouvre une fenêtre qui rejoue les parties sauvegardées, avec les tuiles
colorées du vrai 2048. Si plusieurs parties sont sélectionnées, elles sont
rejouées l'une après l'autre.

Commandes (dans la fenêtre) :
    Espace        lecture / pause
    flèches ← →   reculer / avancer d'une image (en pause)
    flèches ↑ ↓   accélérer / ralentir
    N / P         partie suivante / précédente
    R             recommencer la partie courante
    Échap / Q     quitter

Exemples (depuis la racine du projet) :
    python scripts/replay_gui.py 0            # rejoue la partie d'index 0
    python scripts/replay_gui.py all          # toutes, à la suite
    python scripts/replay_gui.py recordings/run_xxx
    python scripts/replay_gui.py --list       # liste les parties disponibles
"""

from __future__ import annotations

import argparse
import os
import sys

# Ajoute la racine du projet au chemin d'import (cf. scripts/evaluate_random.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from game.renderer import BoardRenderer
from game.replay import (
    DEFAULT_RECORDINGS_DIR,
    list_replays,
    load_replay,
    resolve_replay_paths,
)

DEFAULT_FPS = 6  # images de jeu par seconde au démarrage


def print_listing(directory: str) -> None:
    """Affiche la liste des parties enregistrées disponibles."""
    replays = list_replays(directory)
    if not replays:
        print(f"Aucune partie enregistrée dans '{directory}'.")
        print("Génère-en avec : python scripts/watch_games.py")
        return
    print(f"Parties enregistrées dans '{directory}' :")
    for i, path in enumerate(replays):
        meta = load_replay(path).get("metadata", {})
        print(f"  [{i}] {os.path.basename(path):<24} score={meta.get('score', '?')}")


class ReplayApp:
    """Petite application pygame qui rejoue une liste de parties enregistrées."""

    def __init__(self, paths: list[str], fps: int = DEFAULT_FPS) -> None:
        """Charge les parties et prépare la fenêtre.

        Args:
            paths (list[str]): fichiers de parties à rejouer.
            fps (int): vitesse initiale (images par seconde).
        """
        self.games = [(os.path.basename(p), load_replay(p)) for p in paths]
        self.renderer = BoardRenderer()
        self.fps = fps

        self.game_idx = 0  # partie courante
        self.frame_idx = 0  # image courante dans la partie
        self.playing = True
        self.running = True
        self._accumulator = 0.0  # temps écoulé depuis la dernière image (s)

    @property
    def frames(self) -> list[dict]:
        """Frames de la partie courante."""
        return self.games[self.game_idx][1]["frames"]

    def _clamp_game(self, idx: int) -> None:
        """Change de partie en bornant l'index et en remettant l'image à 0."""
        self.game_idx = max(0, min(idx, len(self.games) - 1))
        self.frame_idx = 0
        self.playing = True

    def _handle_event(self, event: pygame.event.Event) -> None:
        """Traite un événement clavier/fenêtre."""
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT:
                self.frame_idx = min(self.frame_idx + 1, len(self.frames) - 1)
                self.playing = False
            elif event.key == pygame.K_LEFT:
                self.frame_idx = max(self.frame_idx - 1, 0)
                self.playing = False
            elif event.key == pygame.K_UP:
                self.fps = min(self.fps + 2, 60)
            elif event.key == pygame.K_DOWN:
                self.fps = max(self.fps - 2, 1)
            elif event.key == pygame.K_n:
                self._clamp_game(self.game_idx + 1)
            elif event.key == pygame.K_p:
                self._clamp_game(self.game_idx - 1)
            elif event.key == pygame.K_r:
                self.frame_idx = 0
                self.playing = True

    def _advance(self, dt: float) -> None:
        """Fait avancer la lecture automatique selon le temps écoulé."""
        if not self.playing:
            return
        self._accumulator += dt
        step = 1.0 / self.fps
        while self._accumulator >= step:
            self._accumulator -= step
            if self.frame_idx < len(self.frames) - 1:
                self.frame_idx += 1
            else:
                self.playing = False  # fin de la partie : on s'arrête sur la dernière image
                break

    def _draw(self, surface: pygame.Surface) -> None:
        """Dessine l'image courante avec ses informations."""
        name, data = self.games[self.game_idx]
        frame = self.frames[self.frame_idx]
        total = len(self.frames) - 1
        state = "lecture" if self.playing else "pause"
        title = f"{name}  ({self.game_idx + 1}/{len(self.games)})"
        subtitle = (
            f"score {frame.get('score', 0)}  |  {self.frame_idx}/{total}  |  {state} x{self.fps}"
        )
        self.renderer.draw(surface, frame["grid"], title=title, subtitle=subtitle)

    def run(self) -> None:
        """Boucle principale pygame."""
        pygame.init()
        screen = pygame.display.set_mode((self.renderer.width, self.renderer.height))
        pygame.display.set_caption("Replay 2048")
        clock = pygame.time.Clock()

        while self.running:
            dt = clock.tick(60) / 1000.0  # secondes depuis la dernière frame d'affichage
            for event in pygame.event.get():
                self._handle_event(event)
            self._advance(dt)
            self._draw(screen)
            pygame.display.flip()

        pygame.quit()


def main() -> None:
    """Point d'entrée : parse les arguments puis lance la relecture graphique."""
    parser = argparse.ArgumentParser(description="Relecture graphique de parties de 2048.")
    parser.add_argument("targets", nargs="*", help="index(es), fichier(s), dossier(s) ou 'all'")
    parser.add_argument("--dir", default=DEFAULT_RECORDINGS_DIR, help="dossier des enregistrements")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="vitesse initiale (défaut 6)")
    parser.add_argument("-l", "--list", action="store_true", help="liste les parties disponibles")
    args = parser.parse_args()

    if args.list or not args.targets:
        print_listing(args.dir)
        return

    paths = resolve_replay_paths(args.targets, args.dir)
    if not paths:
        print("Aucune partie à rejouer.")
        return

    ReplayApp(paths, fps=max(1, args.fps)).run()


if __name__ == "__main__":
    main()
