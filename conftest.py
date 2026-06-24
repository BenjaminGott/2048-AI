"""Configuration pytest à la racine du projet.

Sa simple présence indique à pytest que la racine du dépôt est le répertoire
racine des tests : pytest ajoute alors ce dossier au chemin d'import, ce qui
permet d'écrire `import game` dans les tests sans bricolage de PYTHONPATH.
"""
