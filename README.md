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

print_all_metrics()

['column_mean_aucroc',
 'column_mean_logloss',
 'weighted_mae',
 'alaska_weighted_auc',
 'mask_accuracy',
 'GAP_vector',
 'AvgSpearman']
 
metric = weighted_auc()
learn = Learner(data, arch, metrics= [metric] )
```



