"""Tests du sélecteur d'agent (agents/loader.py).

On teste surtout `read_algo` (lecture de meta.json), rapide et sans dépendance
lourde. L'instanciation effective des agents est couverte par leurs propres
tests.
"""

from __future__ import annotations

import json
import os
import sys

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.loader import read_algo


def _write_meta(directory: str, version: str, algo: str | None) -> None:
    """Écrit un meta.json minimal pour une version (algo omis si None)."""
    vdir = os.path.join(directory, version)
    os.makedirs(vdir, exist_ok=True)
    meta = {"version": version}
    if algo is not None:
        meta["algo"] = algo
    with open(os.path.join(vdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def test_read_algo_maskable(tmp_path) -> None:
    """read_algo renvoie l'algo écrit dans meta.json."""
    _write_meta(str(tmp_path), "v1", "MaskablePPO")
    assert read_algo("v1", str(tmp_path)) == "MaskablePPO"


def test_read_algo_defaults_to_ppo(tmp_path) -> None:
    """Sans champ 'algo' ou sans meta.json, on retombe sur 'PPO'."""
    _write_meta(str(tmp_path), "v1", None)  # meta sans 'algo'
    assert read_algo("v1", str(tmp_path)) == "PPO"
    # Version inexistante : 'PPO' par défaut, pas d'erreur.
    assert read_algo("v999", str(tmp_path)) == "PPO"
