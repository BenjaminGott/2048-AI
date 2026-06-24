"""Tests unitaires de l'enregistrement/relecture (game/replay.py).

Lancer les tests (depuis la racine du projet) :
    pytest
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.replay import (
    GameRecorder,
    list_replays,
    load_replay,
    parse_save_selection,
    resolve_replay_paths,
)


def test_recorder_metadata() -> None:
    """Les métadonnées reflètent le nombre de coups, le score et la tuile max."""
    rec = GameRecorder()
    rec.capture(np.array([[2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), score=0)
    rec.capture(
        np.array([[0, 0, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
        score=4,
        action=2,
    )

    meta = rec.metadata()
    assert meta["steps"] == 1  # 2 frames -> 1 coup
    assert meta["score"] == 4
    assert meta["max_tile"] == 4


def test_save_and_load_roundtrip(tmp_path) -> None:
    """Sauvegarder puis recharger une partie restitue exactement les frames."""
    rec = GameRecorder()
    grid0 = np.array([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    grid1 = np.array([[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]])
    rec.capture(grid0, score=0)
    rec.capture(grid1, score=4, action=2)

    path = rec.save(directory=str(tmp_path), name="partie_test")
    assert path.endswith(".json")
    assert os.path.isfile(path)

    data = load_replay(path)
    assert data["metadata"]["score"] == 4
    assert len(data["frames"]) == 2
    # La grille rechargée correspond bien à celle enregistrée.
    assert data["frames"][1]["grid"] == grid1.tolist()
    assert data["frames"][1]["action"] == 2
    assert data["frames"][0]["action"] is None


def test_list_replays(tmp_path) -> None:
    """list_replays trouve les fichiers .json (et ignore le reste)."""
    # Dossier vide -> liste vide.
    assert list_replays(str(tmp_path)) == []

    rec = GameRecorder()
    rec.capture(np.zeros((4, 4), dtype=int), score=0)
    rec.save(directory=str(tmp_path), name="a")
    rec.save(directory=str(tmp_path), name="b")

    replays = list_replays(str(tmp_path))
    assert len(replays) == 2
    assert all(p.endswith(".json") for p in replays)


def test_list_replays_missing_dir() -> None:
    """Un dossier inexistant renvoie une liste vide (pas d'erreur)."""
    assert list_replays("dossier_qui_n_existe_pas_12345") == []


def test_list_replays_is_recursive(tmp_path) -> None:
    """list_replays trouve les parties rangées dans des sous-dossiers."""
    rec = GameRecorder()
    rec.capture(np.zeros((4, 4), dtype=int), score=0)
    rec.save(directory=str(tmp_path / "run_1"), name="a")
    rec.save(directory=str(tmp_path / "run_2"), name="b")

    assert len(list_replays(str(tmp_path))) == 2


def test_parse_save_selection() -> None:
    """L'interprétation des réponses du prompt de sauvegarde est correcte."""
    assert parse_save_selection("all", 3) == [0, 1, 2]
    assert parse_save_selection("none", 3) == []
    assert parse_save_selection("", 3) == []
    assert parse_save_selection("0 2", 3) == [0, 2]
    assert parse_save_selection("2,1", 3) == [2, 1]  # ordre conservé
    assert parse_save_selection("0 9 1", 3) == [0, 1]  # indices hors limites ignorés
    assert parse_save_selection("1 1 2", 3) == [1, 2]  # doublons ignorés


def test_resolve_replay_paths(tmp_path) -> None:
    """resolve_replay_paths gère index, 'all', chemins et doublons."""
    rec = GameRecorder()
    rec.capture(np.zeros((4, 4), dtype=int), score=0)
    p0 = rec.save(directory=str(tmp_path), name="g0")
    p1 = rec.save(directory=str(tmp_path), name="g1")

    directory = str(tmp_path)
    assert resolve_replay_paths(["all"], directory) == [p0, p1]
    assert resolve_replay_paths(["0"], directory) == [p0]
    assert resolve_replay_paths(["1", "1"], directory) == [p1]  # dédupliqué
    assert resolve_replay_paths([p0], directory) == [p0]  # chemin direct
    assert resolve_replay_paths(["99"], directory) == []  # index hors limites
