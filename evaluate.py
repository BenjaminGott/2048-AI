"""Comparaison visuelle des versions de l'agent PPO dans le terminal.

Lit `models/progress.json` et affiche un tableau comparatif (baseline + chaque
version), ou un graphe ASCII de la progression du score, ou ré-évalue une
version précise en rejouant des parties.

Exemples (depuis la racine du projet) :
    python evaluate.py                 # tableau comparatif de toutes les versions
    python evaluate.py --plot          # graphe ASCII de la progression du score
    python evaluate.py --version v3    # ré-évalue v3 (rejoue des parties)
"""

from __future__ import annotations

import argparse
import os

from agents.loader import make_agent_for
from train import load_progress


def _fmt_int(value: float) -> str:
    """Formate un entier avec un espace comme séparateur de milliers."""
    return f"{int(round(value)):,}".replace(",", " ")


def print_table(progress: dict) -> None:
    """Affiche le tableau comparatif baseline + versions."""
    rows: list[tuple[str, str, str, str, str, str]] = []

    baseline = progress.get("baseline")
    if baseline:
        rows.append(
            (
                "Baseline",
                "—",
                _fmt_int(baseline["mean_score"]),
                _fmt_int(baseline["mean_max_tile"]),
                f"{baseline['pct_256']:.1f}%",
                f"{baseline['pct_512']:.1f}%",
            )
        )

    for v in progress.get("versions", []):
        rows.append(
            (
                v["version"],
                _fmt_int(v["timesteps_total"]),
                _fmt_int(v["mean_score"]),
                _fmt_int(v["mean_max_tile"]),
                f"{v['pct_256']:.1f}%",
                f"{v['pct_512']:.1f}%",
            )
        )

    if not rows:
        print("Aucune donnée. Entraîne d'abord un modèle avec : python train.py")
        return

    headers = ("Version", "Steps", "Score moy", "Tuile moy", "→ 256", "→ 512")
    widths = [
        max(len(headers[c]), max(len(row[c]) for row in rows)) + 2 for c in range(len(headers))
    ]

    def render_row(cells: tuple[str, ...], fill: str = " ") -> str:
        return "║" + "║".join(c.center(widths[i], fill) for i, c in enumerate(cells)) + "║"

    sep_top = "╔" + "╦".join("═" * w for w in widths) + "╗"
    sep_mid = "╠" + "╬".join("═" * w for w in widths) + "╣"
    sep_bot = "╚" + "╩".join("═" * w for w in widths) + "╝"

    title = "Progression de l'agent PPO — 2048 RL"
    total_width = sum(widths) + len(widths) + 1
    print("╔" + "═" * (total_width - 2) + "╗")
    print("║" + title.center(total_width - 2) + "║")
    print(sep_top)
    print(render_row(headers))
    print(sep_mid)
    for row in rows:
        print(render_row(row))
    print(sep_bot)

    versions = progress.get("versions", [])
    if versions:
        best = max(versions, key=lambda v: v["mean_score"])
        total = versions[-1]["timesteps_total"]
        print(f"Meilleure version : {best['version']}  |  Total entraîné : {_fmt_int(total)} steps")


def print_plot(progress: dict) -> None:
    """Affiche un graphe ASCII (barres verticales) du score moyen par version."""
    labels: list[str] = []
    values: list[float] = []

    if progress.get("baseline"):
        labels.append("base")
        values.append(progress["baseline"]["mean_score"])
    for v in progress.get("versions", []):
        labels.append(v["version"])
        values.append(v["mean_score"])

    if not values:
        print("Aucune donnée à tracer. Entraîne d'abord un modèle (python train.py).")
        return

    height = 12
    max_val = max(values) or 1.0
    col_width = max(len(lbl) for lbl in labels) + 2

    print("\nScore moyen par version")
    for level in range(height, 0, -1):
        threshold = max_val * level / height
        # Étiquette de l'axe Y, affichée seulement sur quelques niveaux.
        y_label = f"{int(threshold):>5}" if level % 3 == 0 or level == height else " " * 5
        cells = []
        for value in values:
            mark = "█" if value >= threshold else " "
            cells.append(mark.center(col_width))
        print(f"{y_label} ┤{''.join(cells)}")

    axis = " " * 5 + " └" + "─" * (col_width * len(values))
    print(axis)
    print(" " * 7 + "".join(lbl.center(col_width) for lbl in labels))


def reevaluate_version(version: str, model_dir: str, n_episodes: int) -> None:
    """Recharge une version et rejoue des parties pour la ré-évaluer."""
    model_path = os.path.join(model_dir, version, "model.zip")
    if not os.path.isfile(model_path):
        print(f"Version introuvable : {model_path}")
        return

    agent = make_agent_for(version, model_dir)
    agent.load(version)
    print(f"Ré-évaluation de {version} sur {n_episodes} parties…")
    stats = agent.evaluate(n_episodes=n_episodes)

    print(f"\n=== {version} ===")
    print(f"Score moyen    : {stats['mean_score']:8.0f}  (± {stats['std_score']:.0f})")
    print(f"Score max      : {stats['max_score']:8d}")
    print(f"Tuile max moy. : {stats['mean_max_tile']:8.0f}")
    print(f"Tuile max abs. : {stats['max_tile_reached']:8d}")
    print(f"% → 256        : {stats['pct_256']:7.1f}%")
    print(f"% → 512        : {stats['pct_512']:7.1f}%")
    print(f"% → 1024       : {stats['pct_1024']:7.1f}%")


def main() -> None:
    """Point d'entrée : parse les arguments et affiche la comparaison demandée."""
    parser = argparse.ArgumentParser(description="Compare les versions de l'agent PPO.")
    parser.add_argument("--version", help="ré-évalue une version précise (ex: v3)")
    parser.add_argument("--plot", action="store_true", help="affiche un graphe ASCII")
    parser.add_argument("--eval-episodes", type=int, default=50, help="parties pour --version")
    parser.add_argument("--model-dir", default="models", help="dossier des modèles")
    args = parser.parse_args()

    if args.version:
        reevaluate_version(args.version, args.model_dir, args.eval_episodes)
        return

    progress = load_progress(args.model_dir)
    if args.plot:
        print_plot(progress)
    else:
        print_table(progress)


if __name__ == "__main__":
    main()
