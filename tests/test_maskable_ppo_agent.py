"""Tests de l'agent MaskablePPO (agents/maskable_ppo_agent.py).

Hyperparamètres et tailles réduits pour rester rapides ; `tmp_path` évite
d'écrire dans le vrai `models/`. Ignorés si sb3-contrib n'est pas installé.
"""

from __future__ import annotations

import os
import sys

import pytest

# Ajoute la racine du projet au chemin d'import (cf. tests/test_board.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("sb3_contrib")

from agents.maskable_ppo_agent import MaskablePPOAgent  # noqa: E402

# Petits hyperparamètres : on teste la mécanique, pas la performance.
FAST_HYPERPARAMS = {"n_steps": 64, "batch_size": 16, "n_epochs": 1}
FAST_STEPS = 64


def _make_agent(model_dir: str) -> MaskablePPOAgent:
    """Crée un agent MaskablePPO rapide (2 envs) dans le dossier donné."""
    return MaskablePPOAgent(model_dir=model_dir, n_envs=2, hyperparams=FAST_HYPERPARAMS)


def test_train_creates_model_and_meta(tmp_path) -> None:
    """train() crée model.zip + meta.json avec algo=MaskablePPO."""
    agent = _make_agent(str(tmp_path / "models"))
    model_path = agent.train(FAST_STEPS, "v1")

    assert os.path.isfile(model_path)
    meta_path = os.path.join(str(tmp_path / "models"), "v1", "meta.json")
    assert os.path.isfile(meta_path)

    import json

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["algo"] == "MaskablePPO"
    assert meta["n_envs"] == 2


def test_predict_returns_valid_action(tmp_path) -> None:
    """predict() renvoie une action dans [0, 3] (et légale grâce au masque)."""
    agent = _make_agent(str(tmp_path / "models"))
    agent.train(FAST_STEPS, "v1")
    obs, _info = agent.env.reset()
    action = agent.predict(obs)
    assert action in (0, 1, 2, 3)


def test_evaluate_keys(tmp_path) -> None:
    """evaluate() renvoie toutes les clés attendues, valeurs valides."""
    agent = _make_agent(str(tmp_path / "models"))
    agent.train(FAST_STEPS, "v1")
    stats = agent.evaluate(n_episodes=2)

    for key in ("mean_score", "max_tile_reached", "pct_256", "pct_512", "pct_1024"):
        assert key in stats
    assert stats["mean_score"] >= 0
    assert 0.0 <= stats["pct_256"] <= 100.0
