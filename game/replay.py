"""Enregistrement et relecture de parties de 2048.

Une partie enregistrée = la suite des états (grilles) traversés, plus quelques
métadonnées (score final, meilleure tuile, agent…). On stocke l'état complet
à chaque coup (et non juste la graine + les actions) : c'est volumineux ?
Non — une grille 4×4 est minuscule — et surtout c'est **robuste**, car la
relecture ne dépend pas de reproduire à l'identique l'aléatoire d'apparition
des tuiles. Rejouer = simplement réafficher les grilles enregistrées.

Format de fichier : JSON.
    {
      "metadata": {"steps": ..., "score": ..., "max_tile": ..., "agent": ...},
      "frames": [
        {"action": null, "grid": [[...]], "score": 0},   # état initial
        {"action": 2,    "grid": [[...]], "score": 4},   # après le 1er coup
        ...
      ]
    }
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

# Dossier par défaut où sont rangées les parties enregistrées.
DEFAULT_RECORDINGS_DIR = "recordings"


class GameRecorder:
    """Capture le déroulé d'une partie, coup par coup, pour la rejouer ensuite."""

    def __init__(self) -> None:
        """Initialise un enregistreur vide."""
        self.frames: list[dict[str, Any]] = []

    def capture(self, grid: np.ndarray, score: int, action: int | None = None) -> None:
        """Enregistre l'état courant de la partie.

        À appeler une fois pour l'état initial (action=None), puis après
        chaque coup avec l'action jouée.

        Args:
            grid (np.ndarray): grille 4×4 après le coup.
            score (int): score courant.
            action (int | None): action jouée pour atteindre cet état
                (None pour l'état initial).
        """
        self.frames.append(
            {
                "action": None if action is None else int(action),
                "grid": np.asarray(grid, dtype=int).tolist(),
                "score": int(score),
            }
        )

    def metadata(self) -> dict[str, Any]:
        """Calcule les métadonnées de la partie à partir des frames capturées.

        Returns:
            dict: nombre de coups, score final, meilleure tuile.
        """
        if not self.frames:
            return {"steps": 0, "score": 0, "max_tile": 0}
        last = self.frames[-1]
        max_tile = int(np.asarray(last["grid"]).max())
        # steps = nombre de coups = nombre de frames - 1 (la 1re est l'état initial).
        return {"steps": len(self.frames) - 1, "score": int(last["score"]), "max_tile": max_tile}

    def save(
        self,
        directory: str = DEFAULT_RECORDINGS_DIR,
        name: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Écrit la partie enregistrée sur disque au format JSON.

        Args:
            directory (str): dossier de destination (créé si absent).
            name (str | None): nom du fichier ; généré automatiquement si None.
            extra_metadata (dict | None): métadonnées additionnelles (ex: agent).

        Returns:
            str: le chemin du fichier écrit.
        """
        os.makedirs(directory, exist_ok=True)
        meta = self.metadata()
        if extra_metadata:
            meta.update(extra_metadata)
        meta["created"] = datetime.now().isoformat(timespec="seconds")

        if name is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            name = f"game_{stamp}_score{meta['score']}.json"
        if not name.endswith(".json"):
            name += ".json"

        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "frames": self.frames}, f)
        return path


def load_replay(path: str) -> dict[str, Any]:
    """Charge une partie enregistrée depuis un fichier JSON.

    Args:
        path (str): chemin du fichier.

    Returns:
        dict: contenu {"metadata": ..., "frames": ...}.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_replays(directory: str = DEFAULT_RECORDINGS_DIR) -> list[str]:
    """Liste les parties enregistrées d'un dossier, sous-dossiers compris.

    La recherche est récursive : les parties rangées dans des sous-dossiers de
    run (ex: `recordings/run_20260624_120000/`) sont donc bien trouvées.

    Args:
        directory (str): dossier à explorer.

    Returns:
        list[str]: chemins des fichiers .json trouvés, triés (vide si absent).
    """
    if not os.path.isdir(directory):
        return []
    found: list[str] = []
    for root, _dirs, files in os.walk(directory):
        found.extend(os.path.join(root, f) for f in files if f.endswith(".json"))
    return sorted(found)


def resolve_replay_paths(targets: list[str], directory: str = DEFAULT_RECORDINGS_DIR) -> list[str]:
    """Transforme des arguments CLI en une liste de chemins de fichiers .json.

    Un argument peut être : un index (0, 1, …) dans la liste du dossier, un
    chemin de fichier, un nom de fichier (cherché dans `directory`), un dossier
    (tous ses .json) ou "all" (tout `directory`). Les entrées introuvables sont
    signalées sur la sortie standard et ignorées. Le résultat est dédupliqué.

    Args:
        targets (list[str]): arguments fournis par l'utilisateur.
        directory (str): dossier de recherche par défaut.

    Returns:
        list[str]: chemins de fichiers à rejouer (ordre conservé, sans doublon).
    """
    available = list_replays(directory)
    paths: list[str] = []

    for target in targets:
        if target == "all":
            paths.extend(available)
        elif os.path.isdir(target):
            paths.extend(list_replays(target))
        elif os.path.isfile(target):
            paths.append(target)
        elif target.isdigit():
            idx = int(target)
            if 0 <= idx < len(available):
                paths.append(available[idx])
            else:
                print(f"Index hors limites : {idx} (0..{len(available) - 1})")
        else:
            candidate = os.path.join(directory, target)
            if os.path.isfile(candidate):
                paths.append(candidate)
            else:
                print(f"Introuvable : {target!r}")

    seen: set[str] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def parse_save_selection(answer: str, count: int) -> list[int]:
    """Interprète la réponse de l'utilisateur au prompt de sauvegarde.

    Args:
        answer (str): saisie (ex: "all", "none", "1 3 5", "2,4").
        count (int): nombre de parties disponibles (indices 0..count-1).

    Returns:
        list[int]: indices des parties à sauvegarder (vide = aucune).
    """
    answer = answer.strip().lower()
    if answer in ("", "none", "n", "aucune", "non"):
        return []
    if answer in ("all", "a", "tout", "toutes", "o", "oui"):
        return list(range(count))

    # Liste d'indices séparés par des espaces et/ou des virgules.
    indices: list[int] = []
    for token in answer.replace(",", " ").split():
        if token.isdigit():
            idx = int(token)
            if 0 <= idx < count and idx not in indices:
                indices.append(idx)
    return indices
