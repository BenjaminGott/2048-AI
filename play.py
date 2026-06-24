"""Jeu 2048 jouable dans le terminal au clavier.

Ce script ne contient que la logique d'affichage et de saisie clavier :
toute la mécanique du jeu vient de `game/board.py` (séparation des
responsabilités).

Contrôles :
    - Flèches directionnelles, ou
    - Z/Q/S/D ou W/A/S/D
    - R : rejouer        Q (ou Échap) : quitter

Lancement :
    python play.py
"""

from __future__ import annotations

import os
import sys

from colorama import init as colorama_init

from game.board import DOWN, LEFT, RIGHT, UP, Board
from game.render_text import board_block
from game.replay import GameRecorder

CELL_WIDTH = 6  # largeur intérieure d'une case, ex: "  2048"


def render(board: Board) -> None:
    """Affiche la grille, le score et la meilleure tuile dans le terminal.

    Args:
        board (Board): partie à afficher.
    """
    lines = board_block(board.grid, cell_width=CELL_WIDTH)
    print("\n".join(lines))
    print(f"Score: {board.score}  |  Meilleure tuile: {board.get_max_tile()}")
    print("Flèches / ZQSD pour jouer  —  R: rejouer  —  Q: quitter")


def _clear_screen() -> None:
    """Efface l'écran du terminal (séquence ANSI, sans dépendance OS)."""
    print("\x1b[2J\x1b[H", end="")


# --- Lecture clavier -------------------------------------------------------
# On lit une touche sans attendre "Entrée". L'implémentation diffère entre
# Windows (msvcrt) et les systèmes Unix (termios/tty).


def _read_key() -> str:
    """Lit une touche et retourne une commande normalisée.

    Returns:
        str: l'une de "up", "down", "left", "right", "restart", "quit",
            ou "" si la touche n'est pas reconnue.
    """
    try:
        import msvcrt  # Windows
    except ImportError:
        return _read_key_unix()

    ch = msvcrt.getch()
    # Les flèches arrivent en deux octets : un préfixe puis un code.
    if ch in (b"\x00", b"\xe0"):
        code = msvcrt.getch()
        return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(code, "")
    return _interpret_char(ch.decode("utf-8", errors="ignore"))


def _read_key_unix() -> str:
    """Variante Unix de `_read_key` (termios/tty)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # début possible d'une séquence de flèche
            seq = sys.stdin.read(2)
            return {"[A": "up", "[B": "down", "[D": "left", "[C": "right"}.get(seq, "quit")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return _interpret_char(ch)


def _interpret_char(ch: str) -> str:
    """Traduit un caractère clavier en commande de jeu.

    Args:
        ch (str): caractère saisi.

    Returns:
        str: commande normalisée (voir `_read_key`).
    """
    ch = ch.lower()
    mapping = {
        "z": "up",
        "w": "up",
        "s": "down",
        "q": "quit",
        "a": "left",  # 'q' = quitter ; 'a' = gauche (clavier QWERTY)
        "d": "right",
        "r": "restart",
        "\x1b": "quit",  # Échap
    }
    # Cas particulier : sur clavier AZERTY 'q' sert à aller à gauche.
    # On privilégie ici 'q' = quitter et on garde 'a' pour la gauche.
    return mapping.get(ch, "")


_COMMAND_TO_DIRECTION = {"up": UP, "down": DOWN, "left": LEFT, "right": RIGHT}


def _new_recorder(board: Board) -> GameRecorder:
    """Crée un enregistreur et capture l'état initial de la partie."""
    recorder = GameRecorder()
    recorder.capture(board.grid, board.score, action=None)
    return recorder


def _prompt_save(recorder: GameRecorder) -> None:
    """Propose de sauvegarder la partie courante (si elle a au moins un coup).

    Args:
        recorder (GameRecorder): enregistreur de la partie terminée/quittée.
    """
    # Pas de terminal interactif (ex: test, pipe) ou partie vide -> on ne demande rien.
    if not sys.stdin.isatty() or len(recorder.frames) <= 1:
        return
    try:
        answer = input("\nSauvegarder cette partie ? (o/N) : ").strip().lower()
    except EOFError:  # entrée indisponible
        return
    if answer in ("o", "oui", "y", "yes"):
        path = recorder.save(
            directory=os.path.join("recordings", "play"),
            extra_metadata={"agent": "human"},
        )
        print(f"Partie enregistrée : {path}")
        print(f"Pour la rejouer : python scripts/replay_game.py {path}")


def play() -> None:
    """Boucle de jeu principale : affiche, lit les touches, applique les coups."""
    colorama_init()  # active l'interprétation des codes ANSI sous Windows
    # Le terminal Windows utilise par défaut cp1252, qui ne sait pas afficher
    # les caractères de cadre (┌ ┐ │ …). On force l'UTF-8 si c'est possible.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    board = Board()
    recorder = _new_recorder(board)  # enregistre la partie pour pouvoir la sauver

    while True:
        _clear_screen()
        render(board)

        if board.is_game_over():
            print(f"\n*** GAME OVER ***  Score final : {board.score}")
            _prompt_save(recorder)
            print("R: rejouer  —  Q: quitter")
            cmd = _read_key()
            while cmd not in ("restart", "quit"):
                cmd = _read_key()
            if cmd == "restart":
                board.reset()
                recorder = _new_recorder(board)
                continue
            break

        cmd = _read_key()
        if cmd == "quit":
            break
        if cmd == "restart":
            board.reset()
            recorder = _new_recorder(board)
            continue
        if cmd in _COMMAND_TO_DIRECTION:
            _gained, changed = board.move(_COMMAND_TO_DIRECTION[cmd])
            if changed:  # on n'enregistre que les coups qui modifient la grille
                recorder.capture(board.grid, board.score, action=_COMMAND_TO_DIRECTION[cmd])
        # Toute autre touche est ignorée : la boucle ré-affiche simplement.

    # Sortie en cours de partie (Q) : on propose aussi de sauvegarder.
    if not board.is_game_over():
        _prompt_save(recorder)

    _clear_screen()
    print(f"Merci d'avoir joué ! Score final : {board.score}")


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        print("\nInterrompu. À bientôt !")
