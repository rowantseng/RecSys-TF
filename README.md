# RecSys-TF
Implementations of recommender systems using TensorFlow.

## Installation

```
pip install -q tensorflow-recommenders
pip install -q --upgrade tensorflow-datasets
pip install -q --upgrade scann
```

## Train and Evaluate

The experiments are conducted using the same pre-built `movielens/100k-ratings` dataset. The data distribution is shown as below. The user ID and model title are transformed into embeddings of size 32. In addition, the movie title is vectorized and applied average pooling to maintain the feature dimension.

|            | Train | Validation |
|------------|-------|------------|
| Data       | 80000 | 20000      |
| Batch Size | 2048  | 4096       |

### Single Task

In this task, query and candidate models are trained using different combinations of dense layers. Also, it is discovered that the timestamp information has great help on model predictions.

|                                    | Accuracy@100 |
|------------------------------------|--------------|
| W/O Timestamp, Layer=[32]          | 0.2136       |
| W/  Timestamp, Layer=[32]          | 0.2670       |
| W/  Timestamp, Layer=[128, 64, 32] | 0.2630       |
| W/  Timestamp, Layer=[64, 64]      | 0.2756       |

![singletask](images/singletask.png#1)

In these experiments, model with two dense layers has the best performance on accuracy@100.

### Multiple Tasks

In this task, query model is trained without time information. The query and candidate models are fixed only one dense layer which has 32 nodes. The combined model can be trained on rating and retrieval tasks by setting weights, `ratingWeight` and `retrievalWeight`.

![multitask](images/multitask.png#1)

|                      | Accuracy@100 |
|----------------------|--------------|
| Retrieval Only       | 0.2153       |
| Rating and Retrieval | 0.2157       |

## References

[TensorFlow Recommenders](https://www.tensorflow.org/recommenders/examples/quickstart)