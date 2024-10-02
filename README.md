# KG from Benchmark Data for AutoML

## Utilization

### Recommend initial models by treating a benchmark dataset from Nas-Bench-Graph as unseen.

```python

from torch_geometric.datasets import CitationFull
import pandas as pd
from kgnas.KGNAS import KGNAS

# Initialize the KGNAS class
# The numerical_weight is the weight of the numerical features in evaluating the dataset similarities, by default it is 0.5. The categorical weight is (1 - numerical_weight).
kgnas = KGNAS(numerical_weight=)

kgnas.power = 1/3
kgnas.upper_bound = 0.8
kgnas.bound_frac = 0.5

# recommend_model will analyze the all the models on top_k_datasets and recommend the top_k_model models that have the highest score.
# Parameters to modify: top_k_dataset, top_k_model.

print(kgnas.get_similar_dataset('cora', top_k=5, sim_metric='gower', include_target_dataset=False))

print(kgnas.recommend_model('cora', top_k_dataset=5, top_k_model=5, sim_metric='gower', score_metric='avg', include_target_dataset=False, style='global'))
# The topology is in the key: 'has_struct_topology', and the structure of each layer is in the keys: 'has_struct_1', 'has_struct_2', 'has_struct_3', 'has_struct_4'

# An example returned dataframe is as follows.

'''
        model  weighted_score has_struct_topology has_struct_1 has_struct_2 has_struct_3 has_struct_4
5435   30111        0.473730        [0, 0, 1, 2]          gat          gcn          gcn          gcn
2979   21415        0.470791        [0, 0, 1, 1]          gcn         sage          gcn         arma
5983   31311        0.470407        [0, 0, 1, 2]          gcn         cheb          gcn          gcn
21134  78004        0.470293        [0, 1, 2, 2]         skip          gat          gat         sage
5459   30141        0.469703        [0, 0, 1, 2]          gat          gcn         sage          gcn
'''

```

### Recommend related models to a given model.

```python

from torch_geometric.datasets import CitationFull
import pandas as pd
from kgnas.KGNAS import KGNAS

# Initialize the KGNAS class
# The numerical_weight is the weight of the numerical features in evaluating the dataset similarities, by default it is 0.5. The categorical weight is (1 - numerical_weight).
kgnas = KGNAS()

kgnas.power = 1/3
kgnas.upper_bound = 0.8
kgnas.bound_frac = 0.5

# Given the unseen dataset name ('cora') and the target model hashed index (10000), get_similar_model will use the structure features and the performance distribution on the top_k similar benchmark datasets to unseen dataset to recommend the top_k_model similar models.

# sim_metric can be fixed to l2.

# sim_weights are the weights for the structural similarity and the performance similarity, by default is [structural, performance] = [1, 4].

# The topology is in the key: 'has_struct_topology', and the structure of each layer is in the keys: 'has_struct_1', 'has_struct_2', 'has_struct_3', 'has_struct_4'

print(kgnas.get_similar_model(source_dataset='cora', source_model=10000, top_k_dataset=2, top_k_model=10, sim_metric='l2', sim_weights=[1, 4]))

# An examplar returned dataframe is as follows.

'''
   model  struct_similarity  perf_similarity  similarity has_struct_topology has_struct_1 has_struct_2 has_struct_3 has_struct_4
0  11000                0.8         0.999995    0.959996        [0, 0, 0, 1]          gcn          gat          gat          gat
1  30000                0.8         0.999984    0.959987        [0, 0, 1, 2]          gat          gat          gat          gat
2  10010                0.8         0.999969    0.959975        [0, 0, 0, 1]          gat          gat          gcn          gat
3  50000                0.8         0.999957    0.959966        [0, 1, 1, 1]          gat          gat          gat          gat
'''

```

<!-- ### Recommend initial models given only an unseen dataset not from the benchmark.

Please check the following demo code for using this package:
    
```python
from torch_geometric.datasets import CitationFull
import pandas as pd
from kgnas.KGNAS import KGNAS

# Initialize the KGNAS class
# The numerical_weight is the weight of the numerical features in evaluating the dataset similarities, by default it is 0.5. The categorical weight is (1 - numerical_weight).
kgnas = KGNAS(numerical_weight=1.0)

kgnas.power = 1/3
kgnas.upper_bound = 0.8
kgnas.bound_frac = 0.5

# STEP4: Find the best performing top-k models from the most similar top-k datasets. The returned dataframe contains the information of the models and the datasets.
candidate_df = kgnas.recommend_model('cora', top_k_dataset=5, top_k_model=5, sim_metric='gower', score_metric='avg', include_target_dataset=True)

candidate_model = candidate_df.iloc[0]
print(candidate_model)

# STEP6: Get the similar models of the given model based on the candidate_df from the last step. To make the recommendation more effective, it is recommended to enlarge the candidate pool from the last step.
# The similarity is calculated based on three parts: the structural similarity, the hyperparameters similarity and the model performance similarity, their weights are currently 1:1:4 towards the final similarity.
similar_model_df = kgnas.get_similar_model(candidate_model, candidate_df, topk=5, sim_metric='l2', sim_weights=[1, 1, 4])

print(similar_model_df.head())
``` -->

## Remarks