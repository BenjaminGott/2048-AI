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

import sys

from colorama import Style
from colorama import init as colorama_init

from game.board import DOWN, LEFT, RIGHT, UP, Board

# --- Couleurs des tuiles (codes ANSI 256 couleurs) -------------------------
# On associe chaque valeur de tuile à une couleur de fond + une couleur de
# texte pour la lisibilité. Au-delà de 2048, on réutilise la dernière couleur.

_RESET = Style.RESET_ALL
CELL_WIDTH = 6  # largeur intérieure d'une case, ex: "  2048"

# (couleur de fond, couleur de texte) en codes ANSI 256.
TILE_COLORS: dict[int, tuple[int, int]] = {
    0: (236, 244),  # case vide : gris foncé
    2: (255, 236),  # blanc cassé, texte foncé
    4: (223, 236),  # beige
    8: (215, 235),  # orange clair
    16: (208, 231),  # orange
    32: (202, 231),  # orange-rouge
    64: (196, 231),  # rouge
    128: (227, 236),  # jaune clair
    256: (220, 236),  # jaune
    512: (214, 236),  # jaune-or
    1024: (190, 16),  # vert-jaune
    2048: (46, 16),  # vert vif
}
_MAX_COLOR_KEY = 2048


def _ansi_cell(value: int) -> str:
    """Retourne la représentation colorée d'une case (largeur fixe).

    Args:
        value (int): valeur de la tuile (0 = case vide).

    Returns:
        str: chaîne de longueur CELL_WIDTH, colorée via codes ANSI.
    """
    bg, fg = TILE_COLORS.get(min(value, _MAX_COLOR_KEY), TILE_COLORS[_MAX_COLOR_KEY])
    text = "" if value == 0 else str(value)
    content = text.rjust(CELL_WIDTH)
    return f"\x1b[48;5;{bg}m\x1b[38;5;{fg}m{content}{_RESET}"


def render(board: Board) -> None:
    """Affiche la grille, le score et la meilleure tuile dans le terminal.

    Args:
        board (Board): partie à afficher.
    """
    size = board.grid.shape[0]
    top = "┌" + "┬".join(["─" * CELL_WIDTH] * size) + "┐"
    mid = "├" + "┼".join(["─" * CELL_WIDTH] * size) + "┤"
    bot = "└" + "┴".join(["─" * CELL_WIDTH] * size) + "┘"

    lines: list[str] = [top]
    for r in range(size):
        cells = "│".join(_ansi_cell(int(v)) for v in board.grid[r])
        lines.append("│" + cells + "│")
        lines.append(mid if r < size - 1 else bot)

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


def play() -> None:
    """Boucle de jeu principale : affiche, lit les touches, applique les coups."""
    colorama_init()  # active l'interprétation des codes ANSI sous Windows
    # Le terminal Windows utilise par défaut cp1252, qui ne sait pas afficher
    # les caractères de cadre (┌ ┐ │ …). On force l'UTF-8 si c'est possible.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    board = Board()

    while True:
        _clear_screen()
        render(board)

        if board.is_game_over():
            print(f"\n*** GAME OVER ***  Score final : {board.score}")
            print("R: rejouer  —  Q: quitter")
            cmd = _read_key()
            while cmd not in ("restart", "quit"):
                cmd = _read_key()
            if cmd == "restart":
                board.reset()
                continue
            break

        cmd = _read_key()
        if cmd == "quit":
            break
        if cmd == "restart":
            board.reset()
            continue
        if cmd in _COMMAND_TO_DIRECTION:
            board.move(_COMMAND_TO_DIRECTION[cmd])
        # Toute autre touche est ignorée : la boucle ré-affiche simplement.

    _clear_screen()
    print(f"Merci d'avoir joué ! Score final : {board.score}")


if __name__ == "__main__":
    try:
        play()
    except KeyboardInterrupt:
        print("\nInterrompu. À bientôt !")
