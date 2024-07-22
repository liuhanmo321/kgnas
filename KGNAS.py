import pandas as pd 
import torch 
import torch_geometric
import numpy as np
import networkx as nx
import os
import json

from .dataset.DatasetDescription import DatasetDescription
from .model.ModelDescription import ModelDescription
from nas_bench_graph import Arch

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

    def calculate_model_similarity(self, source_model_data: pd.Series, candidate_df: pd.DataFrame, sim_metric='gower', sim_weights=[1, 1, 4]):

        # print(source_model_data)
        source_model_data = pd.Series(source_model_data)
        source_model_data['model'] = 'source_model'

        # Integrate the data together
        temp_df = candidate_df.copy(deep=True)
        temp_df = pd.concat([temp_df, source_model_data.to_frame().T])

        # Process the numerical data and categorical data
        for col in temp_df.columns:
            if col in ['model', 'dataset'] + self.model_desc.relation_names['hardware']:
                continue
            
            if temp_df[col].dtype == 'object':
                temp_df[col] = (temp_df[col] == source_model_data[col]).astype(float)
            else:
                temp_df[col] = self.process_numerical_data(temp_df[col])

        temp_df.set_index('model', inplace=True)

        # Calculate the similarity
        source_hyper_param_vector = temp_df.loc['source_model'][self.model_desc.relation_names['hyper_param']].astype(float)
        source_structure_vector = temp_df.loc['source_model'][self.model_desc.relation_names['structure']].astype(float)
        
        temp_df['hyper_param_similarity'] = 1
        temp_df['struct_similarity'] = 1
        for model in temp_df.index:
            candidate_hyper_param_vector = temp_df.loc[model][self.model_desc.relation_names['hyper_param']].astype(float)
            temp_df.loc[model, 'hyper_param_similarity'] = self.pairwise_similarity(source_hyper_param_vector, candidate_hyper_param_vector, sim_metric=sim_metric)
            
            candidate_structure_vector = temp_df.loc[model][self.model_desc.relation_names['structure']].astype(float)
            temp_df.loc[model, 'structure_similarity'] = self.pairwise_similarity(source_structure_vector, candidate_structure_vector, sim_metric=sim_metric)

        temp_df.drop('source_model', axis=0, inplace=True)

        temp_df['perf_similarity'] = 1

        models = temp_df.index
        temp_df['structure_id'] = [model[1:] for model in models]
        temp_perf_df = self.model_desc.hyper_relation_info.copy(deep=True)
        temp_perf_df['target_entity'] = temp_perf_df['target_entity'].apply(lambda x: x[1:])
        perf_df = temp_perf_df[['target_entity', 'source_entity', 'perf']].pivot(index='target_entity', columns='source_entity', values='perf')
        perf_df.reset_index(inplace=True)
        perf_df.set_index('target_entity', inplace=True)
        perf_df.fillna(0, inplace=True)

        struct_list = [int(i) for i in source_model_data['has_struct_topology'][1:-1].split(',')]
        layer_list = [source_model_data[f'has_struct_{i}'] for i in range(1, 5)]
        source_struct_id = str(Arch(struct_list, layer_list).valid_hash())
        for model in temp_df.index:
            temp_df.loc[model, 'perf_similarity'] = self.pairwise_similarity(perf_df.loc[temp_df.loc[model, 'structure_id']], perf_df.loc[source_struct_id], sim_metric=sim_metric)

        # Averate the similarities as the final similarity
        temp_df['similarity'] = (sim_weights[0] * temp_df['hyper_param_similarity'] + sim_weights[1] * temp_df['structure_similarity'] + sim_weights[2] * temp_df['perf_similarity']) / sum(sim_weights)

        # Process the data for future usage
        temp_df.drop('structure_id', axis=1, inplace=True)
        temp_df.sort_values(by='similarity', ascending=False, inplace=True)
        temp_df.reset_index(inplace=True)
        
        # Return only the necessary information
        return temp_df


    def get_similar_model(self, source_model_data: pd.Series, candidate_df: pd.DataFrame, topk=5, sim_metric='gower', sim_weights=[1, 1, 1]):
        model_similarity_df = self.calculate_model_similarity(source_model_data, candidate_df, sim_metric=sim_metric, sim_weights=sim_weights)
        
        return model_similarity_df.head(topk)
    
    def get_similar_dataset(self, target_dataset_name, top_k=5, sim_metric='gower'):
        dataset_similarities_df = self.calculate_dataset_similarity(target_dataset_name, sim_metric=sim_metric)

        return dataset_similarities_df.head(top_k)
    
    def recommend_model(self, target_dataset_name, top_k_dataset=5, top_k_model=5, score_metric='avg'):
        recommend_model_df = pd.DataFrame()

        dataset_similarities_df = self.get_similar_dataset(target_dataset_name, top_k=top_k_dataset)

        # Obtain the top-k models for each of the similar datasets with their performance information.
        for _, row in dataset_similarities_df.iterrows():
            temp_top_model_df = self.get_top_model_from_dataset(row['dataset'], top_k=top_k_model)
            temp_top_model_df['dataset'] = row['dataset']
            temp_top_model_df['dataset_similarity'] = row['dataset_similarity']
            if recommend_model_df.shape[0] == 0:
                recommend_model_df = temp_top_model_df.copy(deep=True)
            else:
                recommend_model_df = pd.concat([recommend_model_df, temp_top_model_df])

        recommend_model_df['model'] = recommend_model_df['model'].apply(lambda x: str(x))

        # Join binary information
        for relation in self.model_desc.two_ary_info['relation'].unique():
            temp_df = self.model_desc.two_ary_info[self.model_desc.two_ary_info['relation'] == relation][['source_entity', 'target_entity']].rename(columns={'source_entity': 'model', 'target_entity': relation})
            recommend_model_df = pd.merge(recommend_model_df, temp_df, left_on='model', right_on='model', how='left')
        
        # Join uniary information
        recommend_model_df = recommend_model_df.join(self.model_desc.uniary_info.set_index('model'), on='model', how='left')

        # Calculate the hybrid score combining the similarity and performances for comprehensive recommendation.
        dataset_similarity = recommend_model_df['dataset_similarity'].to_numpy()
        dataset_similarity = (dataset_similarity - dataset_similarity.min()) / (dataset_similarity.max() - dataset_similarity.min())
        
        perf = recommend_model_df['perf'].to_numpy()
        perf = (perf - perf.min()) / (perf.max() - perf.min())
        
        if score_metric == 'avg':
            recommend_model_df['score'] = (dataset_similarity + perf) / 2
        if score_metric == 'mult':
            recommend_model_df['score'] = dataset_similarity * perf
        
        return recommend_model_df

    def get_top_model_from_dataset(self, dataset_name, perf_metric='perf', top_k=10):
        top_model_df = self.model_desc.hyper_relation_info[self.model_desc.hyper_relation_info['source_entity'] == dataset_name].sort_values(by=perf_metric, ascending=False).head(top_k).copy(deep=True)
        top_model_df = top_model_df[['target_entity', perf_metric]]
        top_model_df.sort_values(by=perf_metric, ascending=False, inplace=True)
        top_model_df.rename(columns={'target_entity': 'model'}, inplace=True)

        return top_model_df



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

        target_vector = target_vector.to_numpy()
        bench_vector = bench_vector.to_numpy()

        if sim_metric == 'cosine':
            return torch.nn.functional.cosine_similarity(torch.tensor(target_vector), torch.tensor(bench_vector), dim=0).item()
        if sim_metric == 'l2':
            return 1 - torch.nn.functional.mse_loss(torch.tensor(target_vector), torch.tensor(bench_vector)).item()
        if sim_metric == 'l1':
            return 1 - torch.nn.functional.l1_loss(torch.tensor(target_vector), torch.tensor(bench_vector)).item()