# kaggle fastai custom metrics
***

Custom Metrics for fastai v1 for kaggle competitions.

***
## Disclaimer :

Each Kaggle competition has a unique metrics suited to its, need this package lets you download those custom metrics to be used with fastai library.
Since Metrics are an important part to evaulate your models performance.

## Installation 

```sh
pip install kaggle-fastai-custom-metrics==1.0.1
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
```

```
+------------------------------------------------------------------------+---------------------+
|                            Competition URL                             |        Metric       |
+------------------------------------------------------------------------+---------------------+
|          https://www.kaggle.com/c/plant-pathology-2020-fgvc7           | column_mean_logloss |
| https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge |  column_mean_aucroc |
|         https://www.kaggle.com/c/trends-assessment-prediction          |     weighted_mae    |
|          https://www.kaggle.com/c/alaska2-image-steganalysis           | alaska_weighted_auc |
|            https://www.kaggle.com/c/google-quest-challenge             |     AvgSpearman     |
|           https://www.kaggle.com/c/landmark-recognition-2020           |      GAP_vector     |
+------------------------------------------------------------------------+---------------------+
```
```
metric = weighted_auc()
learn = Learner(data, arch, metrics= [metric] )
```



