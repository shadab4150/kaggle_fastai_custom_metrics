# kaggle fastai custom metrics
***

Custom Metrics for fastai v1 for kaggle competitions.

***
## Disclaimer :

Each Kaggle competition has a unique metrics suited to its, need this package lets you download those custom metrics to be used with fastai library.
Since Metrics are an important part to evaulate your models performance.

## Installation 

```sh
pip install kaggle_fastai_custom_metrics
```

or

```bash
git clone https://github.com/shadab4150/kaggle_fastai_custom_metrics
cd kaggle_fastai_custom_metrics
pip install .
```
## Usage :

```python
from kaggle_fastai_custom_metrics.kfcm import *
metric = weighted_auc()
learn = Learner(data, arch, metrics= [metric] )
```



