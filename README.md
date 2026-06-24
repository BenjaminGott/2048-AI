# 2048-RL — Apprentissage par renforcement sur le 2048

Projet d'apprentissage : entraîner une IA à jouer au 2048 avec du
Reinforcement Learning (PPO / MaskablePPO via `stable-baselines3`), avec des
outils pour jouer, visualiser, enregistrer et rejouer les parties.

> Détails de conception et avancement : voir `.claude/CLAUDE.md`.

---

## 1. Installation

```bash
# (optionnel) créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# dépendances d'exécution
pip install -r requirements.txt

# dépendances de développement (tests, lint, hooks) — recommandé
pip install -r requirements-dev.txt
pre-commit install               # active les hooks de formatage au commit
```

Toutes les commandes se lancent **depuis la racine du projet**.

---

## 2. Jouer / visualiser (pas besoin d'entraînement)

| Commande | Description |
|---|---|
| `python play.py` | Jouer soi-même au clavier (terminal). Touches : flèches ou Z/Q/S/D, `R` rejouer, `Q` quitter. Propose de sauvegarder la partie à la fin. |
| `python scripts/watch_games.py` | Regarder l'agent aléatoire jouer **6 parties en simultané** (terminal). |
| `python scripts/watch_games.py -n 12 -c 4 -d 0.05` | 12 parties, 4 par ligne, 0,05 s entre deux coups. |
| `python scripts/watch_games.py --record` | Idem + sauvegarde automatique de toutes les parties dans `recordings/`. |

À la fin de `play.py` et `watch_games.py`, un prompt propose de sauvegarder une / plusieurs / toutes les parties.

---

## 3. Baseline (agent aléatoire)

```bash
python scripts/evaluate_random.py            # stats sur 100 parties
python scripts/evaluate_random.py 500        # sur 500 parties
```

Affiche score moyen, meilleure tuile et taux d'atteinte (256 / 512…). Sert de **référence à battre**.

---

## 4. Entraîner l'agent (PPO / MaskablePPO)

```bash
# MaskablePPO par défaut (masquage des coups invalides + envs parallèles)
python train.py --timesteps 100000 --versions 5 --eval-episodes 50

# reprendre l'entraînement après la dernière version (warm start)
python train.py --resume --versions 3

# PPO simple (sans masquage), comme point de comparaison
python train.py --no-mask

# régler le nombre d'environnements parallèles
python train.py --n-envs 8
```

| Argument | Défaut | Rôle |
|---|---|---|
| `--timesteps` | 100000 | pas d'entraînement par version |
| `--versions` | 5 | nombre de versions entraînées à la suite |
| `--eval-episodes` | 50 | parties jouées pour évaluer chaque version |
| `--resume` | — | reprend depuis la dernière version sauvegardée |
| `--no-mask` | — | utilise PPO simple au lieu de MaskablePPO |
| `--n-envs` | 8 | environnements parallèles (MaskablePPO) |
| `--model-dir` | `models` | dossier de sauvegarde des modèles |

Chaque version est sauvegardée dans `models/vN/` (`model.zip`, `best_model.zip`, `meta.json`, `eval.json`) et agrégée dans `models/progress.json`.

---

## 5. Comparer / évaluer les versions

```bash
python evaluate.py                  # tableau comparatif (baseline + toutes les versions)
python evaluate.py --plot           # graphe ASCII de la progression du score
python evaluate.py --version v3     # ré-évalue v3 en rejouant des parties
```

---

## 6. Enregistrer et rejouer des parties

```bash
# lister les parties enregistrées
python scripts/replay_game.py --list

# rejouer dans le TERMINAL (couleurs ANSI)
python scripts/replay_game.py 0          # la partie d'index 0
python scripts/replay_game.py 0 2 3      # plusieurs en simultané
python scripts/replay_game.py all        # toutes
python scripts/replay_game.py recordings/run_xxx   # un dossier entier

# rejouer en GRAPHIQUE (fenêtre pygame)
python scripts/replay_gui.py all
python scripts/replay_gui.py 0 --fps 8
```

---

## 7. Regarder le modèle entraîné jouer EN DIRECT (graphique)

```bash
python scripts/play_gui.py             # dernière version entraînée
python scripts/play_gui.py v3          # une version précise
python scripts/play_gui.py --list      # versions disponibles
python scripts/play_gui.py v5 --fps 8  # plus rapide
```

Commandes dans la fenêtre : **Espace** lecture/pause · **→** un coup (en pause) · **↑/↓** vitesse · **R** nouvelle partie · **Échap/Q** quitter.

---

## 8. Tests & qualité du code

```bash
pytest                       # toute la suite de tests
pytest tests/test_env.py -v  # un fichier en mode détaillé

ruff format .                # formater le code
ruff check . --fix           # linter (et corriger)
pre-commit run --all-files   # tous les hooks sur tout le projet
```

La CI GitHub Actions (`.github/workflows/tests.yml`) lance Ruff puis `pytest`
(Python 3.10 → 3.13) à chaque push et pull request.

---

## 9. Carte du projet (qui fait quoi)

```
game/        logique du jeu (board), rendu texte/pygame, enregistrement (replay)
env/         environnement Gymnasium (Game2048Env, action masking)
agents/      random_agent, ppo_agent, maskable_ppo_agent, loader
scripts/     evaluate_random, watch_games, replay_game, replay_gui, play_gui
train.py     entraînement versionné         evaluate.py  comparaison des versions
play.py      jeu manuel au clavier          tests/       tests unitaires
models/      modèles entraînés (non versionné)   recordings/  parties enregistrées (non versionné)
```
