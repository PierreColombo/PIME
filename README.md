# PIME: Python Information Measures Estimation

Documentation (not necessarily up to date with master branch): https://pime.readthedocs.io/en/latest/

To-do list: https://www.notion.so/305cd6b61831453db1e0f6b52b113d81?v=9dfa802a5b4f4e5f97e789f1ef6d0356
`

Pour Moi il y a deux parties: pour les measures de similarité


Discrete Discrete (ce que j'ai codé dans InfoLM)
 - Entre deux measues:
    - F divergences
    - Fisher Rao
    - LP distances
 - Sur une seule measure
    - Entropy

Continue Discrete (ce que Malik a codé dans son TIM):


Continue Continue:
  - Entre deux measues:
    -  Mutual information
    -  Les closes formes gaussiennes
  - Sur une seule measure:
    - Entropy  


###  Running tests locally

```bash
pip install pytest pytest-cov black isort

# Make sure pip install -e . has been run
isort pime
black pime

py.test --cov-report term --cov=pime ./unit_tests     
```