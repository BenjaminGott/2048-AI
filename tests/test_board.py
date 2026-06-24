"""Tests unitaires de la logique du jeu (game/board.py).

Lancer les tests (depuis la racine du projet) :
    pytest
ou simplement :
    python tests/test_board.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ajoute la racine du projet au chemin d'import pour que `import game`
# fonctionne aussi bien via pytest que via `python tests/test_board.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import DOWN, LEFT, RIGHT, UP, Board


def _make_board(grid: list[list[int]]) -> Board:
    """Construit un Board dont la grille est fixée (pour des tests déterministes).

    On contourne le tirage aléatoire de `__init__` en réécrivant la grille
    et en remettant le score à zéro après coup.
    """
    board = Board()
    board.grid = np.array(grid, dtype=np.int32)
    board.score = 0
    return board


def test_slide_left_merges_pair() -> None:
    """Un mouvement gauche sur [2,2,0,0] doit donner [4,0,0,0]."""
    board = _make_board(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    gained, changed = board.move(LEFT)

    assert changed is True
    assert gained == 4
    # La fusion 2+2 a bien produit un 4 collé à gauche.
    assert board.grid[0, 0] == 4
    # Après un coup valide, exactement une nouvelle tuile apparaît : la grille
    # contient donc le 4 fusionné + 1 nouvelle tuile (qui peut tomber n'importe
    # où, y compris en [0, 1] -> on ne teste donc pas une case précise vide).
    non_zero = board.grid[board.grid != 0]
    assert non_zero.size == 2


def test_slide_left_no_triple_merge() -> None:
    """[2,2,2,0] vers la gauche fusionne UNE seule paire -> [4,2,0,0]."""
    board = _make_board(
        [
            [2, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    gained, changed = board.move(LEFT)

    assert changed is True
    assert gained == 4
    assert board.grid[0, 0] == 4
    assert board.grid[0, 1] == 2


def test_invalid_move_does_not_change_grid() -> None:
    """Un mouvement invalide laisse la grille inchangée et ne gagne rien."""
    # Toutes les tuiles sont déjà collées à gauche et non fusionnables :
    # un mouvement vers la gauche ne change donc rien.
    board = _make_board(
        [
            [2, 4, 0, 0],
            [4, 2, 0, 0],
            [2, 4, 0, 0],
            [4, 2, 0, 0],
        ]
    )
    before = board.grid.copy()
    gained, changed = board.move(LEFT)

    assert changed is False
    assert gained == 0
    assert np.array_equal(board.grid, before)


def test_game_over_detection() -> None:
    """Une grille pleine sans fusion possible doit être détectée comme finie."""
    # Damier de valeurs alternées : aucune paire adjacente identique.
    board = _make_board(
        [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]
    )
    assert board.is_game_over() is True
    assert board.get_valid_moves() == []


def test_game_not_over_when_move_exists() -> None:
    """Une grille avec une paire fusionnable n'est pas terminée."""
    board = _make_board(
        [
            [2, 2, 4, 8],
            [4, 8, 16, 32],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
        ]
    )
    assert board.is_game_over() is False
    # Au moins le mouvement gauche/droite (fusion des deux 2) est valide.
    assert len(board.get_valid_moves()) > 0


def test_move_up_stacks_column() -> None:
    """Un mouvement haut empile et fusionne une colonne correctement."""
    board = _make_board(
        [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    gained, changed = board.move(UP)

    assert changed is True
    assert gained == 4
    # Colonne 0 : 2+2 -> 4, puis le 4 existant -> [4, 4, 0, 0].
    assert board.grid[0, 0] == 4
    assert board.grid[1, 0] == 4


def test_move_down_stacks_column() -> None:
    """Un mouvement bas empile les tuiles vers le bas."""
    board = _make_board(
        [
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    gained, changed = board.move(DOWN)

    assert changed is True
    assert gained == 4
    # Le 4 fusionné doit se retrouver tout en bas de la colonne 0.
    assert board.grid[3, 0] == 4


def test_move_right_merges_pair() -> None:
    """[0,0,2,2] vers la droite donne [0,0,0,4]."""
    board = _make_board(
        [
            [0, 0, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    gained, changed = board.move(RIGHT)

    assert changed is True
    assert gained == 4
    # La fusion s'est faite à droite ; une nouvelle tuile apparaît ailleurs
    # (potentiellement en [0, 2]), donc on vérifie le total plutôt qu'une case.
    assert board.grid[0, 3] == 4
    non_zero = board.grid[board.grid != 0]
    assert non_zero.size == 2


def test_reset_returns_grid_with_two_tiles() -> None:
    """Après reset, la grille contient exactement deux tuiles (2 ou 4)."""
    board = Board()
    grid = board.reset()

    non_empty = grid[grid != 0]
    assert non_empty.size == 2
    assert board.score == 0
    assert all(v in (2, 4) for v in non_empty)


def test_invalid_direction_raises() -> None:
    """Une direction hors {0,1,2,3} lève une ValueError."""
    board = Board()
    try:
        board.move(42)
    except ValueError:
        pass
    else:  # pragma: no cover - le test échoue si aucune exception
        raise AssertionError("move(42) aurait dû lever ValueError")


if __name__ == "__main__":
    # Exécution directe sans pytest : on lance chaque test et on affiche le bilan.
    import sys

    tests = [
        obj for name, obj in sorted(globals().items()) if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
        except AssertionError as exc:  # noqa: PERF203
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[ERR]  {test.__name__}: {exc!r}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests réussis.")
    sys.exit(1 if failures else 0)
