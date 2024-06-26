[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.12.2](https://img.shields.io/badge/Python-3.12.2-green)](https://www.python.org/downloads/release/python-3122/)

# Clusterkolommen

## Benodigdheden

[![Python 3.12.2](https://img.shields.io/badge/Python-3.12.2-green)](https://www.python.org/downloads/release/python-3122/)

[![pipenv >= 2023.4.20](https://img.shields.io/badge/pipenv-%3E%3D2023.4.20-blue)](https://pypi.org/project/pipenv/)

## Installatie

Zorg dat een versie van Python (>= 3.7) is geïnstalleerd. Je kan Pipenv 
vervolgens installeren met:

```sh
python -m pip install --user pipenv
```

`cd` vervolgens naar de root van dit project en voer dan de volgende command 
uit:

```sh
python -m pipenv install
```

Dit zal automatisch een virtual environment met Python 3.12 aanmaken. Vervolgens
zullen alle dependencies hier naartoe gedownload worden.

Activeer de Python 3.12 omgeving door de volgende command uit te voeren:

```sh
python -m pipenv shell
```

Alles zou nu gereed moeten zijn voor gebruik.

## Algoritme

Het algoritme bestaat uit twee onderdelen: `ClusteringModel` en `Elbow`

### src/algorithm/clustering.py

In deze module staat het clusteringsmodel `ClusteringModel`. Om deze te 
gebruiken heb je een 2-dimensionale numpy array nodig.

```py
example_data = np.array([
    [1,2,3], 
    [1,2,3], 
    [1,2,3],
])

num_clusters = 2

model = ClusteringModel(example_dataset, num_clusters, "lloyds")
model.tain(example_data)

model.cluster_centers
model.error
```

`ClusteringModel.cluster_centers` geeft de clustercentra terug en 
`ClusteringModel.error` geeft de score (WCSS) van het model terug.
