"""Tests unitaires de l'agent PPO (agents/ppo_agent.py) et du versioning.

Pour rester rapides, ces tests utilisent des hyperparamètres réduits (peu de
pas) et la fixture `tmp_path` de pytest afin de ne jamais écrire dans le vrai
dossier `models/`. Si stable-baselines3 n'est pas installé, ils sont ignorés.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("stable_baselines3")

from agents.ppo_agent import PPOAgent  # noqa: E402  (après importorskip, voulu)
from env.game2048_env import Game2048Env  # noqa: E402
from train import update_progress  # noqa: E402

# Hyperparamètres minuscules : on veut juste vérifier la mécanique, pas apprendre.
FAST_HYPERPARAMS = {"n_steps": 64, "batch_size": 16, "n_epochs": 1}
FAST_STEPS = 64


def _make_agent(model_dir: str) -> PPOAgent:
    """Crée un agent PPO rapide dans le dossier donné."""
    return PPOAgent(Game2048Env(), model_dir=model_dir, hyperparams=FAST_HYPERPARAMS)


def test_instantiation(tmp_path) -> None:
    """L'agent s'instancie et crée le dossier des modèles."""
    agent = _make_agent(str(tmp_path / "models"))
    assert agent.model is not None
    assert os.path.isdir(str(tmp_path / "models"))


def test_train_creates_files(tmp_path) -> None:
    """train() crée model.zip et meta.json dans models/{version}/."""
    agent = _make_agent(str(tmp_path / "models"))
    model_path = agent.train(FAST_STEPS, "vtest")

    assert os.path.isfile(model_path)
    assert model_path.endswith(os.path.join("vtest", "model.zip"))
    meta_path = os.path.join(str(tmp_path / "models"), "vtest", "meta.json")
    assert os.path.isfile(meta_path)

    import json

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["version"] == "vtest"
    assert "hyperparams" in meta and "gamma" in meta["hyperparams"]


def test_evaluate_returns_expected_keys(tmp_path) -> None:
    """evaluate() renvoie toutes les clés attendues avec des valeurs valides."""
    agent = _make_agent(str(tmp_path / "models"))
    agent.train(FAST_STEPS, "v1")
    stats = agent.evaluate(n_episodes=3)

    expected = {
        "mean_score",
        "std_score",
        "max_score",
        "mean_max_tile",
        "max_tile_reached",
        "pct_256",
        "pct_512",
        "pct_1024",
    }
    assert expected.issubset(stats.keys())
    assert stats["mean_score"] >= 0
    assert 0.0 <= stats["pct_256"] <= 100.0
    assert stats["max_tile_reached"] >= 2


def test_load_then_predict(tmp_path) -> None:
    """Après load(), predict() renvoie une action valide dans [0, 3]."""
    model_dir = str(tmp_path / "models")
    agent = _make_agent(model_dir)
    agent.train(FAST_STEPS, "v1")

    other = _make_agent(model_dir)
    other.load("v1")
    obs, _info = other.env.reset()
    action = other.predict(obs)
    assert action in (0, 1, 2, 3)


def test_progress_updates_over_two_versions(tmp_path) -> None:
    """progress.json contient bien deux entrées triées après deux versions."""
    model_dir = str(tmp_path / "models")
    os.makedirs(model_dir, exist_ok=True)
    fake_stats = {
        "mean_score": 1500.0,
        "std_score": 300.0,
        "max_score": 3000,
        "mean_max_tile": 128.0,
        "max_tile_reached": 256,
        "pct_256": 10.0,
        "pct_512": 1.0,
        "pct_1024": 0.0,
    }

    update_progress(model_dir, "v1", 1000, fake_stats)
    progress = update_progress(model_dir, "v2", 2000, fake_stats)

    versions = progress["versions"]
    assert [v["version"] for v in versions] == ["v1", "v2"]
    assert versions[1]["timesteps_total"] == 2000
    assert "baseline" in progress
    assert os.path.isfile(os.path.join(model_dir, "progress.json"))
