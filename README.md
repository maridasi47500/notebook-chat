# HJupytext Project

## Description
Ce projet utilise `jupytext` pour convertir des scripts Python en notebooks Jupyter et vice versa. Cela facilite le développement collaboratif et la gestion du code versionné.

## Installation
Pour installer les dépendances nécessaires, suivez les étapes ci-dessous :

```sh
git clone https://github.com/votre-utilisateur/hjupytext-project.git
cd hjupytext-project
conda env create -f environment.yml
conda activate hjupytext-env
pip install -r requirements.txt
Convertir un script Python en notebook Jupyter
Pour convertir un script Python (script.py) en notebook Jupyter (script.ipynb), utilisez la commande suivante :

sh
jupytext --to notebook script.py
Convertir un notebook Jupyter en script Python
Pour convertir un notebook Jupyter (notebook.ipynb) en script Python (notebook.py), utilisez la commande suivante :

sh
jupytext --to py notebook.ipynb
Exécution et visualisation des notebooks
Pour ouvrir et exécuter les notebooks, lancez Jupyter Notebook :

sh
jupyter notebook
# notebook-chat
