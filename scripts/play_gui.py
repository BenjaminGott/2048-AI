"""Visualisation GRAPHIQUE en direct d'un modèle entraîné qui joue au 2048.

Ouvre une fenêtre pygame où l'agent (PPO ou MaskablePPO) joue une partie en
temps réel, avec les tuiles colorées du vrai 2048. C'est l'équivalent de
`replay_gui.py`, mais le modèle joue en direct au lieu de rejouer un fichier.

Commandes (dans la fenêtre) :
    Espace        lecture / pause
    flèche →      avance d'un coup (en pause)
    flèches ↑ ↓   accélérer / ralentir
    R             nouvelle partie
    Échap / Q     quitter

Exemples (depuis la racine du projet) :
    python scripts/play_gui.py              # dernière version entraînée
    python scripts/play_gui.py v3           # une version précise
    python scripts/play_gui.py --list       # liste les versions disponibles
    python scripts/play_gui.py v5 --fps 4   # plus lent
"""

from __future__ import annotations

import argparse
import os
import sys

# Ajoute la racine du projet au chemin d'import (cf. scripts/evaluate_random.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame

from agents.loader import make_agent_for, read_algo
from game.renderer import BoardRenderer
from train import list_versions

DEFAULT_FPS = 4  # coups par seconde au démarrage


class AgentPlayApp:
    """Fenêtre pygame où un modèle chargé joue des parties en direct."""

    def __init__(self, version: str, model_dir: str, fps: int = DEFAULT_FPS) -> None:
        """Charge le modèle de la version donnée et prépare la partie.

        Args:
            version (str): version à charger (ex: "v5").
            model_dir (str): dossier des modèles.
            fps (int): vitesse initiale (coups par seconde).
        """
        self.version = version
        self.algo = read_algo(version, model_dir)
        self.agent = make_agent_for(version, model_dir)
        self.agent.load(version)
        self.env = self.agent.env  # env (éventuellement masqué) que l'agent pilote

        self.renderer = BoardRenderer()
        self.fps = fps
        self.running = True
        self._accumulator = 0.0
        self._new_game()

    # --- État de la partie -------------------------------------------------

    def _new_game(self) -> None:
        """Démarre une nouvelle partie."""
        self.obs, self.info = self.env.reset()
        self.terminated = False
        self.steps = 0
        self.playing = True
        self._accumulator = 0.0

    @property
    def _grid(self):
        """Grille courante (on traverse l'éventuel wrapper jusqu'au Board)."""
        return self.env.unwrapped.board.grid

    def _step_once(self) -> None:
        """Fait jouer un coup à l'agent."""
        if self.terminated:
            return
        action = self.agent.predict(self.obs)
        self.obs, _reward, self.terminated, _truncated, self.info = self.env.step(action)
        self.steps += 1

    # --- Boucle pygame -----------------------------------------------------

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                self.running = False
            elif event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.key == pygame.K_RIGHT:
                self.playing = False
                self._step_once()
            elif event.key == pygame.K_UP:
                self.fps = min(self.fps + 1, 30)
            elif event.key == pygame.K_DOWN:
                self.fps = max(self.fps - 1, 1)
            elif event.key == pygame.K_r:
                self._new_game()

    def _advance(self, dt: float) -> None:
        if not self.playing or self.terminated:
            return
        self._accumulator += dt
        step = 1.0 / self.fps
        while self._accumulator >= step:
            self._accumulator -= step
            self._step_once()
            if self.terminated:
                break

    def _draw(self, surface: pygame.Surface) -> None:
        state = "FIN" if self.terminated else ("lecture" if self.playing else "pause")
        title = f"{self.version} · {self.algo}"
        subtitle = (
            f"score {self.info.get('score', 0)}  |  {self.steps} coups  |  {state} x{self.fps}"
        )
        self.renderer.draw(surface, self._grid, title=title, subtitle=subtitle)

    def run(self) -> None:
        """Boucle principale pygame."""
        pygame.init()
        screen = pygame.display.set_mode((self.renderer.width, self.renderer.height))
        pygame.display.set_caption("2048 — agent en direct")
        clock = pygame.time.Clock()

        while self.running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                self._handle_event(event)
            self._advance(dt)
            self._draw(screen)
            pygame.display.flip()

        pygame.quit()


def resolve_version(version: str | None, model_dir: str) -> str | None:
    """Détermine la version à charger (la dernière par défaut)."""
    versions = list_versions(model_dir)
    if not versions:
        return None
    if version is None:
        return versions[-1]
    return version if version in versions else None


def main() -> None:
    """Point d'entrée : parse les arguments puis lance la visualisation live."""
    parser = argparse.ArgumentParser(description="Regarde un modèle 2048 jouer en direct.")
    parser.add_argument("version", nargs="?", help="version à charger (défaut : la dernière)")
    parser.add_argument("--model-dir", default="models", help="dossier des modèles")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="coups/seconde (défaut 4)")
    parser.add_argument("-l", "--list", action="store_true", help="liste les versions disponibles")
    args = parser.parse_args()

    versions = list_versions(args.model_dir)
    if args.list or not versions:
        if versions:
            print(f"Versions disponibles dans '{args.model_dir}' : {', '.join(versions)}")
        else:
            print(f"Aucune version dans '{args.model_dir}'. Entraîne d'abord : python train.py")
        return

    version = resolve_version(args.version, args.model_dir)
    if version is None:
        print(f"Version introuvable : {args.version!r}. Disponibles : {', '.join(versions)}")
        return

    AgentPlayApp(version, args.model_dir, fps=max(1, args.fps)).run()


if __name__ == "__main__":
    main()
