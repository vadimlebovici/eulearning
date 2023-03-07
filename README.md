`Eulearning`: Euler characteristic tools for topological data analysis
============

**Authors**: Olympio Hacquard and Vadim Lebovici

# Description
`eulearning` is a Python package to compute Euler characteristic profiles of multi-parameter filtrations, as well as their Radon and hybrid transforms. <!-- It is based on the article  --> Please find short usage demonstrations in the notebooks of the `demos/` folder.

- `descriptors.py` contains the scikit-learn transformers computing Euler characteristic tools.

- `datasets.py` contains all datasets used in our article, at the exception of graph datasets which come from [here](https://networkrepository.com/) and can be downloaded at the right format from the [Perslay repository](https://github.com/MathieuCarriere/perslay). One graph dataset is included for a demo. 

- `utils.py` contains auxilary but necessary functions. For instance, in contains a way to compute multi-parameter filtrations in the form of vectorized simplex trees.

# Package requirements

**For Euler characteristic descriptors**:
[numpy](https://numpy.org/),
[scikit-learn](https://scikit-learn.org/stable/).

**For multi-parameter filtrations**:
[numba](https://numba.pydata.org/),
[scipy](https://scipy.org/),
[GraphRicciCurvature](https://github.com/saibalmars/GraphRicciCurvature).

**For datasets**:
[Gudhi](https://gudhi.inria.fr/python/latest/index.html),
[tadasets](https://pypi.org/project/tadasets/). Moreover, we use code from Guillaume Moroz's repository [dpp](https://gitlab.inria.fr/gmoro/point_process/) to generate Ginibre and Poisson point clouds.

**For demos**: 
[xgboost](https://xgboost.readthedocs.io/en/stable/), 
[matplotlib](https://matplotlib.org/).