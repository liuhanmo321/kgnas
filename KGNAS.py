import pandas as pd 
import torch 
import torch_geometric
import numpy as np
import networkx as nx
import os
import json

from .dataset.DatasetDescription import DatasetDescription
from .model.ModelDescription import ModelDescription

BENCH_DATASET_NAME = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']

class KGNAS:
    def __init__(self, kg_dir='./KG/', numerical_weight=0.5):
        self.dataset_desc = DatasetDescription()
        self.model_desc = ModelDescription()
        self.KG = nx.Graph()

        self.kg_dir = kg_dir

        self.entities = []

        self.numerical_weight = numerical_weight
        self.categorical_weight = 1 - numerical_weight

    def calculate_dataset_similarity(self, target_dataset_name, sim_metric='gower'):
        # uniary similarity: a vector of decimal numbers for each of the statistical values.
        temp_df = self.dataset_desc.uniary_info.copy(deep=True)
        numerical_columns = temp_df.select_dtypes(include=['number']).columns

        for col in numerical_columns:
            temp_df[col] = self.process_numerical_data(temp_df[col])

        # two-ary similarity: a vector of 0s and 1s.
        categorical_columns = self.dataset_desc.two_ary_info['relation'].unique()
        for relation in categorical_columns:
            temp_df[relation] = 0

            relation_df = self.dataset_desc.two_ary_info[self.dataset_desc.two_ary_info['relation'] == relation]
            target_entity = relation_df[relation_df['source_entity'] == target_dataset_name]['target_entity'].values[0]
            temp_df[relation] = (relation_df['target_entity'] == target_entity).astype(int)

        dataset_similarities = {'dataset': [], 'dataset_similarity': []}
        temp_df.set_index('Dataset', inplace=True)
        target_vector = temp_df.loc[target_dataset_name]
        for dataset in BENCH_DATASET_NAME:
            bench_vector = temp_df.loc[dataset]
            dataset_similarities['dataset'].append(dataset)
            dataset_similarities['dataset_similarity'].append(self.pairwise_similarity(target_vector, bench_vector, sim_metric=sim_metric, numerical_columns=numerical_columns, categorical_columns=categorical_columns))

        # print(dataset_similarities)
        
        dataset_similarities_df = pd.DataFrame(dataset_similarities)
        dataset_similarities_df.sort_values(by=['dataset_similarity'], ascending=False, inplace=True)

        return dataset_similarities_df

    def calculate_model_similarity(self):
        pass
    
    def get_similar_dataset(self, target_dataset_name, top_k=5, sim_metric='gower'):
        dataset_similarities_df = self.calculate_dataset_similarity(target_dataset_name, sim_metric=sim_metric)

        return dataset_similarities_df.head(top_k)
    
    def recommend_model(self, target_dataset_name, top_k_dataset=5, top_k_model=5):
        recommend_model_df = pd.DataFrame()

        dataset_similarities_df = self.get_similar_dataset(target_dataset_name, top_k=top_k_dataset)
        # print(dataset_similarities_df.shape)
        # print(dataset_similarities_df.head())

        for _, row in dataset_similarities_df.iterrows():
            # print(row)
            temp_top_model_df = self.get_top_model_from_dataset(row['dataset'], top_k=top_k_model)
            temp_top_model_df['dataset'] = row['dataset']
            temp_top_model_df['dataset_similarity'] = row['dataset_similarity']
            if recommend_model_df.shape[0] == 0:
                recommend_model_df = temp_top_model_df.copy(deep=True)
            else:
                recommend_model_df = pd.concat([recommend_model_df, temp_top_model_df])

        recommend_model_df['model'] = recommend_model_df['model'].apply(lambda x: str(x))

        for relation in self.model_desc.two_ary_info['relation'].unique():
            temp_df = self.model_desc.two_ary_info[self.model_desc.two_ary_info['relation'] == relation][['source_entity', 'target_entity']].rename(columns={'source_entity': 'model', 'target_entity': relation})
            recommend_model_df = pd.merge(recommend_model_df, temp_df, left_on='model', right_on='model', how='left')
        
        recommend_model_df = recommend_model_df.join(self.model_desc.uniary_info.set_index('model'), on='model', how='left')
        
        return recommend_model_df

    def get_top_model_from_dataset(self, dataset_name, perf_metric='perf', top_k=10):
        top_model_df = self.model_desc.hyper_relation_info[self.model_desc.hyper_relation_info['source_entity'] == dataset_name].sort_values(by=perf_metric, ascending=False).head(top_k).copy(deep=True)
        top_model_df = top_model_df[['target_entity', perf_metric]]
        top_model_df.sort_values(by=perf_metric, ascending=False, inplace=True)
        top_model_df.rename(columns={'target_entity': 'model'}, inplace=True)

        return top_model_df

    def get_similar_model(self):
        pass

    def visualize(self):
        pass

    def generate_knowledge_graph(self):
        model_2ary_edges = [(row['source_entity'], row['target_entity'], {'relation': row['relation']}) for idx, row in self.model_desc.two_ary_info.iterrows()]
        # print(len(model_2ary_edges))
        self.KG.add_edges_from(model_2ary_edges)

        data_2ary_edges = [(row['source_entity'], row['target_entity'], {'relation': row['relation']}) for idx, row in self.dataset_desc.two_ary_info.iterrows()]
        # print(len(data_2ary_edges))
        # print(self.dataset_desc.two_ary_info['source_entity'].unique())
        self.KG.add_edges_from(data_2ary_edges)
        
        hyper_relation_names = list(self.model_desc.hyper_relation_info.columns.difference(['source_entity', 'target_entity']))
        model_hyper_edges = [(row['source_entity'], row['target_entity'], {key: row[key] for key in hyper_relation_names}) for idx, row in self.model_desc.hyper_relation_info.iterrows()]
        # print(self.model_desc.hyper_relation_info['source_entity'].unique())
        self.KG.add_edges_from(model_hyper_edges)

        self.entities = set(self.KG.nodes)

    def save_knowledge_graph(self, dir='./KG/'):
        if not os.path.exists(dir):
            os.makedirs(dir)

        data = nx.node_link_data(self.KG)
        with open(dir+"KGNAS.json", 'w') as f:
            json.dump(data, f)
        # nx.write_weighted_edgelist(self.KG, dir+"KGNAS.weighted.edgelist")

    def load_knowledge_graph(self, dir='./KG/'):
        # self.KG = nx.read_weighted_edgelist(dir+"KGNAS.weighted.edgelist")
        with open(dir+"KGNAS.json", 'r') as f:
            data = json.load(f)
            self.KG = nx.node_link_graph(data)

    def get_knowledge_graph(self):
        pass

    def summrize_knowledge_graph(self):
        if not os.path.exists(self.kg_dir+"KGNAS.json"):
            self.generate_knowledge_graph()
        else:
            self.load_knowledge_graph(self.kg_dir)
            self.entities = set(self.KG.nodes)
    
        num_nodes = self.KG.number_of_nodes()
        num_edges = self.KG.number_of_edges()
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")

        degree_sequence = [d for n, d in self.KG.degree()]
        average_degree = sum(degree_sequence) / num_nodes
        max_degree = max(degree_sequence)
        min_degree = min(degree_sequence)

        print("Average degree:", average_degree)
        print("Maximum degree:", max_degree)
        print("Minimum degree:", min_degree)

    def add_dataset_description(self, dataset_name, dataset=None, semantic_description=None, root_dir='datasets/', num_samples=20, num_hops=2, seed=42):
        self.dataset_desc.add_description(dataset_name, dataset=dataset, semantic_description=semantic_description, root_dir=root_dir, num_samples=num_samples, num_hops=num_hops, seed=seed)

    def process_numerical_data(self, column):
        column = column.fillna(0)
        column = (column - column.min()) / (column.max() - column.min())
        return column
    
    def pairwise_similarity(self, target_vector, bench_vector, sim_metric='gower', numerical_columns=None, categorical_columns=None):
        if sim_metric == 'cosine':
            return torch.nn.functional.cosine_similarity(torch.tensor(target_vector.to_numpy()), torch.tensor(bench_vector.to_numpy())).item()
        if sim_metric == 'l2':
            return 1 - torch.nn.functional.mse_loss(torch.tensor(target_vector.to_numpy()), torch.tensor(bench_vector.to_numpy())).item()
        if sim_metric == 'l1':
            return 1 - torch.nn.functional.l1_loss(torch.tensor(target_vector.to_numpy()), torch.tensor(bench_vector.to_numpy())).item()
        if sim_metric == 'gower':

            target_num_vector = target_vector[numerical_columns].to_numpy()
            target_cat_vector = target_vector[categorical_columns].to_numpy()

            bench_num_vector = bench_vector[numerical_columns].to_numpy()
            bench_cat_vector = bench_vector[categorical_columns].to_numpy()

            def dice_similarity(a, b):
                """
                Dice distance between to 0/1 vectors

                params:
                a (np.array): the first binary vector
                b (np.array): the second binary vector

                return:
                float: Dice distance
                """
                assert np.array_equal(a, a.astype(bool)), "Must be binary vector (0/1)"
                assert np.array_equal(b, b.astype(bool)), "Must be binary vector (0/1)"

                # Compute the intersections and the nums of 0s and 1s
                intersection = np.sum(a * b)
                a_sum = np.sum(a)
                b_sum = np.sum(b)

                # Compute the Dice distance
                dice_similarity = (2.0 * intersection) / (a_sum + b_sum)
                # dice_distance = 1 - dice_similarity

                return dice_similarity
            
            numerical_similarity = np.mean(1 - np.abs(target_num_vector - bench_num_vector))
            categorical_similarity = dice_similarity(target_cat_vector, bench_cat_vector)
            
            similarity = self.numerical_weight * numerical_similarity + self.categorical_weight * categorical_similarity
            
            return similarity