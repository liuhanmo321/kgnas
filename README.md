# KG from NAS for AutoML

## Utilization

### Recommend initial models given only an unseen dataset

Please check the following demo code for using this package:
    
```python

from torch_geometric.datasets import CitationFull
import pandas as pd
from kgnas.KGNAS import KGNAS

# Initialize the KGNAS class
# The numerical_weight is the weight of the numerical features in evaluating the dataset similarities, by default it is 0.5. The categorical weight is (1 - numerical_weight).
kgnas = KGNAS(numerical_weight=0.5)

# If you need to check the characteristics of the KG, you can use the following functions.
# It transforms all the 2-ary information into a graph using networkx and calculates the essential information of the graph.
# kgnas.summrize_knowledge_graph()

# STEP1: Create the dataset in the format of PyG (like following I load the dataset directly)
dataset = CitationFull(root='./datasets/', name='DBLP')

# STEP2: Fill in the related information of the dataset, currently only the listed info are requried. The suggested options are:
# {
#    "Dataset": Name of the dataset,
#    "NumClasses": intiger value,
#     "Metric": ["Accuracy", "ROC-AUC"], How the model performance is evaluated.
#     "Task": ["Node Classification", "Edge Prediction"], What is the task of the dataset.
#     "Type": ["Homogeneous", "Heterogeneous"], What is the type of the dataset.
#     "LabelType": ["Single", "Multiple", "None"], Whether one node has one label or multiple labels or no label.
#     "FeatureType": ["Node", "Edge", "Both"], Whether the features are node features or edge features or both.
#     "EdgeSemantic": ["Citation", "Coauthor", "Coappearance", "Biological Interaction"], What does the edge imply, like citation, coauthor, etc. If not included in existings ones, you can add your own, which is also helpful for the recommendation.
#     "Domain": ["Computer Science", "Medical", "Physics", "Business"], What is the domain of the dataset, like computer science, medical, etc. If not included in existings ones, you can add your own, which is also helpful for the recommendation.
# }
semantic_description = {
    "Dataset": "DBLP",
    "NumClasses": 4,
    "Metric": "Accuracy",
    "Task": "Node Classification",
    "Type": "Homogeneous",
    "LabelType": "Single",
    "FeatureType": "Node",
    "EdgeSemantic": "Citation",
    "Domain": "Computer Science",
}

# STEP3: Add the dataset to the knowledge graph. The first index of Dataset from the PyG is the actual dataset, so dataset[0] is passed.
kgnas.add_dataset_description('DBLP', dataset[0], semantic_description=semantic_description)

# STEP4: The similar datasets of the given dataset can be obtained by calling get similar dataset, which gives the dataset names and the similarities.
# The numerical features of the data set is standardized to [0, 1] and the categorical values are transformed into 0s and 1s, where 0 means different from the given dataset and 1 means same as the given dataset.
# The similarity can be calculated by: cosine similarity (cosine), l1 distance (l1), l2 distance (l2), gower distance (gower). The recommended metric is gower distance, as it separately handles the numerical and categorical values.
print(kgnas.get_similar_dataset('DBLP', top_k=5, sim_metric='gower'))

# STEP5: The recommended models can be obtained directly by calling the following function, the top_k_dataset and top_k_model are the number of similar datasets and models to be recommended.
# This returns a dataframe with the all the related information of the models and the datasets.
print(kgnas.recommend_model('DBLP', top_k_dataset=5, top_k_model=5))
```

### Recommend renewed models given the similar datasets to the new dataset and the existing model.

```python

from torch_geometric.datasets import CitationFull
import pandas as pd
from kgnas.KGNAS import KGNAS

# Initialize the KGNAS class
# The numerical_weight is the weight of the numerical features in evaluating the dataset similarities, by default it is 0.5. The categorical weight is (1 - numerical_weight).
kgnas = KGNAS(numerical_weight=0.5)

# If you need to check the characteristics of the KG, you can use the following functions.
# It transforms all the 2-ary information into a graph using networkx and calculates the essential information of the graph.
# kgnas.summrize_knowledge_graph()

# STEP1: Create the dataset in the format of PyG (like following I load the dataset directly)
dataset = CitationFull(root='./datasets/', name='DBLP')

# STEP2: Fill in the related information of the dataset, currently only the listed info are requried. The suggested options are:
# {
#    "Dataset": Name of the dataset,
#    "NumClasses": intiger value,
#     "Metric": ["Accuracy", "ROC-AUC"], How the model performance is evaluated.
#     "Task": ["Node Classification", "Edge Prediction"], What is the task of the dataset.
#     "Type": ["Homogeneous", "Heterogeneous"], What is the type of the dataset.
#     "LabelType": ["Single", "Multiple", "None"], Whether one node has one label or multiple labels or no label.
#     "FeatureType": ["Node", "Edge", "Both"], Whether the features are node features or edge features or both.
#     "EdgeSemantic": ["Citation", "Coauthor", "Coappearance", "Biological Interaction"], What does the edge imply, like citation, coauthor, etc. If not included in existings ones, you can add your own, which is also helpful for the recommendation.
#     "Domain": ["Computer Science", "Medical", "Physics", "Business"], What is the domain of the dataset, like computer science, medical, etc. If not included in existings ones, you can add your own, which is also helpful for the recommendation.
# }
semantic_description = {
    "Dataset": "DBLP",
    "NumClasses": 4,
    "Metric": "Accuracy",
    "Task": "Node Classification",
    "Type": "Homogeneous",
    "LabelType": "Single",
    "FeatureType": "Node",
    "EdgeSemantic": "Citation",
    "Domain": "Computer Science",
}

# STEP3: Add the dataset to the knowledge graph. The first index of Dataset from the PyG is the actual dataset, so dataset[0] is passed.
kgnas.add_dataset_description('DBLP', dataset[0], semantic_description=semantic_description)

# STEP4: Find the best performing top-k models from the most similar top-k datasets. The returned dataframe contains the information of the models and the datasets.
candidate_df = kgnas.recommend_model('DBLP', top_k_dataset=3, top_k_model=20)

# STEP5: Establish the model that you want to improve on. Here, as an example, the model is selected as the first model in the candidate_df.
# In practice, you can first copy one model in the candidate_df and renew the content by the actual model to save effort. An example is shown below.
# model                                                    147206
# perf                                                   0.715667
# dataset                                                citeseer
# dataset_similarity                                     0.923063
# has_struct_topology                                [0, 0, 1, 3] (string)
# has_struct_1                                                 fc
# has_struct_2                                                gin
# has_struct_3                                                gat
# has_struct_4                                              graph
# has_CPU                Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
# has_GPU                             NVIDIA Tesla V100 with 16GB
# has_OperatingSystem               CentOS Linux release 7.6.1810
# has_Optimizer                                               SGD
# NumPrevLayer                                                  0
# NumPostLayer                                                  1
# Dimension                                                   256
# Dropout                                                     0.7
# LR                                                          0.2
# WD                                                       0.0005
# NumEpoch                                                    400
# score                                                  0.549145
# The important column names to modify are: has_struct_topology, has_struct_1, has_struct_2, has_struct_3, has_struct_4, has_Optimizer, NumPrevLayer, NumPostLayer, Dimension, Dropout, LR, WD, NumEpoch.
# You can ignore the rest of the columns, as they are not counted towards the model similarity.
candidate_model = candidate_df.iloc[0]
print(candidate_model)

# STEP6: Get the similar models of the given model based on the candidate_df from the last step. To make the recommendation more effective, it is recommended to enlarge the candidate pool from the last step.
# The similarity is calculated based on three parts: the structural similarity, the hyperparameters similarity and the model performance similarity, their weights are 1:1:4 towards the final similarity.
similar_model_df = kgnas.get_similar_model(candidate_model, candidate_df, topk=5, sim_metric='l2')

print(similar_model_df.head())
```


## Remarks

1. Currently only support the initial model recommendation for the given dataset. The model-model similarity is not considered yet, so that no further operations avaliable.
2. The existing model information are not able to be added into the KG, will be considered later.