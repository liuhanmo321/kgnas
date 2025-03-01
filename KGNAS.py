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
from .utils.utils import *
from .model.HashDecoder import HashDecoder

BENCH_DATASET_NAME = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']

class KGNAS:
    """
    Class representing the KGNAS (Knowledge Graph Neural Architecture Search) module.

    Args:
        kg_dir (str): Directory path to the knowledge graph.
        numerical_weight (float): Weight assigned to numerical features in the KG. Default is 0.5.

    Attributes:
        dataset_desc (DatasetDescription): Instance of the DatasetDescription class.
        model_desc (ModelDescription): Instance of the ModelDescription class.
        KG (nx.Graph): Knowledge graph.

        kg_dir (str): Directory path to the knowledge graph.
        entities (list): List of entities in the knowledge graph.

        numerical_weight (float): Weight assigned to numerical features in the KG.
        categorical_weight (float): Weight assigned to categorical features in the KG.

    """

    def __init__(self, kg_dir='./KG/', numerical_weight=1.0, load=False):
        self.dataset_desc = DatasetDescription(kg_dir=kg_dir, load=load)
        self.model_desc = ModelDescription(kg_dir=kg_dir, load=load)
        self.dataset_model_desc = pd.DataFrame()
        self.KG = nx.Graph()

        self.kg_dir = kg_dir

        self.entities = []

        self.numerical_weight = numerical_weight
        self.categorical_weight = 1 - numerical_weight

        self.normalize = True
        self.standardize = False
        self.activation = 'gower'
        self.power = 1
        self.upper_bound = 1.0
        self.lower_bound = 0
        self.process_method = 'normal'
        self.bound_frac = 1
        self.dynamic_upperbound_dict = {}

        self.normalized_data_desc_df = None
        self.dataset_numerical_columns = None
        self.dataset_categorical_columns = None

    def calculate_dataset_similarity(self, target_dataset_name, sim_metric='gower'):
        """
        Calculate the similarity between the target dataset and other datasets in the KGNAS dataset.

        Args:
            target_dataset_name (str): The name of the target dataset.
            sim_metric (str, optional): The similarity metric to use. Defaults to 'gower'.

        Returns:
            pandas.DataFrame: A DataFrame containing the dataset names and their similarity scores, sorted in descending order.

        """
        # uniary similarity: a vector of decimal numbers for each of the statistical values.
        temp_df = self.dataset_desc.uniary_info.copy(deep=True)
        numerical_columns = temp_df.select_dtypes(include=['number']).columns
        self.dataset_numerical_columns = numerical_columns

        target_index = temp_df[temp_df['dataset'] == target_dataset_name].index[0]

        IQR_list = []
        for col in numerical_columns:
            temp_df[col], IQR = self.process_numerical_data(temp_df[col], process_method=self.process_method, target_index=target_index)
            if IQR is not None:
                IQR_list.append(IQR)
        dynamic_upperbound = np.array(IQR_list) * self.bound_frac if len(IQR_list) > 0 else None

        self.dynamic_upperbound_dict[target_dataset_name] = dynamic_upperbound

        # two-ary similarity: a vector of 0s and 1s.
        categorical_columns = self.dataset_desc.two_ary_info['relation'].unique()
        self.dataset_categorical_columns = categorical_columns
        for relation in categorical_columns:
            temp_df[relation] = 0

            relation_df = self.dataset_desc.two_ary_info[self.dataset_desc.two_ary_info['relation'] == relation]
            target_entity = relation_df[relation_df['source_entity'] == target_dataset_name]['target_entity'].values[0]
            temp_df[relation] = (relation_df['target_entity'] == target_entity).astype(int)

        dataset_similarities = {'dataset': [], 'dataset_similarity': []}
        temp_df.set_index('dataset', inplace=True)

        self.normalized_data_desc_df = temp_df.copy(deep=True)

        target_vector = temp_df.loc[target_dataset_name]
        for dataset in BENCH_DATASET_NAME:
            bench_vector = temp_df.loc[dataset]
            dataset_similarities['dataset'].append(dataset)
            dataset_similarities['dataset_similarity'].append(self.pairwise_similarity(target_vector, bench_vector, sim_metric=sim_metric, numerical_columns=numerical_columns, categorical_columns=categorical_columns, standardize=self.standardize, activation=self.activation, dynamic_upperbound=dynamic_upperbound))
        
        dataset_similarities_df = pd.DataFrame(dataset_similarities)
        dataset_similarities_df.sort_values(by=['dataset_similarity'], ascending=False, inplace=True)

        return dataset_similarities_df

    # def calculate_model_similarity(self, source_model_data: pd.Series, candidate_df: pd.DataFrame, sim_metric='gower', sim_weights=[1, 1, 4]):
    #     """
    #     Calculates the similarity between a source model and a dataframe of candidate models.
    #     The similarity is calculated based on the hyperparameters, structure, and performance of the models.
    #     Three types of similarities are calculated separately and then combined using the weights provided.

    #     Args:
    #         source_model_data (pd.Series): The data of the source model.
    #         candidate_df (pd.DataFrame): The dataframe containing the candidate models.
    #         sim_metric (str, optional): The similarity metric to use. Defaults to 'gower'.
    #         sim_weights (list, optional): The weights for the different similarity components. Defaults to [1, 1, 4].

    #     Returns:
    #         pd.DataFrame: The candidate dataframe with additional similarity information.
    #     """
    #     # print(source_model_data)
    #     source_model_data = pd.Series(source_model_data)
    #     source_model_data['model'] = 'source_model'

    #     # Integrate the data together
    #     temp_df = candidate_df.copy(deep=True)
    #     temp_df = pd.concat([temp_df, source_model_data.to_frame().T])

    #     # Process the numerical data and categorical data
    #     for col in temp_df.columns:
    #         if col in ['model', 'dataset'] + self.model_desc.relation_names['hardware']:
    #             continue
            
    #         if temp_df[col].dtype == 'object':
    #             temp_df[col] = (temp_df[col] == source_model_data[col]).astype(float)
    #         else:
    #             temp_df[col] = self.process_numerical_data(temp_df[col])

    #     temp_df.set_index('model', inplace=True)

    #     # Calculate the similarity
    #     source_hyper_param_vector = temp_df.loc['source_model'][self.model_desc.relation_names['hyper_param']].astype(float)
    #     source_structure_vector = temp_df.loc['source_model'][self.model_desc.relation_names['structure']].astype(float)
        
    #     temp_df['hyper_param_similarity'] = 1
    #     temp_df['struct_similarity'] = 1
    #     for model in temp_df.index:
    #         candidate_hyper_param_vector = temp_df.loc[model][self.model_desc.relation_names['hyper_param']].astype(float)
    #         temp_df.loc[model, 'hyper_param_similarity'] = self.pairwise_similarity(source_hyper_param_vector, candidate_hyper_param_vector, sim_metric=sim_metric)
            
    #         candidate_structure_vector = temp_df.loc[model][self.model_desc.relation_names['structure']].astype(float)
    #         temp_df.loc[model, 'struct_similarity'] = self.pairwise_similarity(source_structure_vector, candidate_structure_vector, sim_metric=sim_metric)

    #     temp_df.drop('source_model', axis=0, inplace=True)

    #     temp_df['perf_similarity'] = 1

    #     models = temp_df.index
    #     temp_df['structure_id'] = [model[1:] for model in models]
    #     temp_perf_df = self.model_desc.hyper_relation_info.copy(deep=True)
    #     temp_perf_df['target_entity'] = temp_perf_df['target_entity'].apply(lambda x: x[1:])
    #     perf_df = temp_perf_df[['target_entity', 'source_entity', 'perf']].pivot(index='target_entity', columns='source_entity', values='perf')
    #     perf_df.reset_index(inplace=True)
    #     perf_df.set_index('target_entity', inplace=True)
    #     perf_df.fillna(0, inplace=True)

    #     struct_list = [int(i) for i in source_model_data['has_struct_topology'][1:-1].split(',')]
    #     layer_list = [source_model_data[f'has_struct_{i}'] for i in range(1, 5)]
    #     source_struct_id = str(Arch(struct_list, layer_list).valid_hash())
    #     for model in temp_df.index:
    #         temp_df.loc[model, 'perf_similarity'] = self.pairwise_similarity(perf_df.loc[temp_df.loc[model, 'structure_id']], perf_df.loc[source_struct_id], sim_metric=sim_metric)

    #     # Averate the similarities as the final similarity
    #     temp_df['similarity'] = (sim_weights[0] * temp_df['hyper_param_similarity'] + sim_weights[1] * temp_df['struct_similarity'] + sim_weights[2] * temp_df['perf_similarity']) / sum(sim_weights)

    #     temp_df.reset_index(inplace=True)
    #     candidate_df['similarity'] = temp_df['similarity'].copy(deep=True)
    #     candidate_df['hyper_param_similarity'] = temp_df['hyper_param_similarity'].copy(deep=True)
    #     candidate_df['struct_similarity'] = temp_df['struct_similarity'].copy(deep=True)
    #     candidate_df['perf_similarity'] = temp_df['perf_similarity'].copy(deep=True)

    #     # Return only the necessary information
    #     return candidate_df

    def get_similar_model(self, source_dataset, source_model, top_k_dataset, top_k_model, sim_metric='gower', sim_weights=[1, 4]):
        """
        Calculates the similarity between a source model and a dataframe of candidate models.
        The similarity is calculated based on the hyperparameters, structure, and performance of the models.
        Three types of similarities are calculated separately and then combined using the weights provided.

        Args:
            source_model: The hashed id of a source_model.
            sim_metric (str, optional): The similarity metric to use. Defaults to 'gower'.
            sim_weights (list, optional): The weights for the different similarity components. Defaults to [1, 1, 4].

        Returns:
            pd.DataFrame: The candidate dataframe with additional similarity information.
        """

        similar_dataset_df = self.get_similar_dataset(source_dataset, top_k=top_k_dataset, sim_metric='gower', include_target_dataset=False)
        similar_datasets = similar_dataset_df['dataset'].to_list()

        model_similarity_df = self.model_desc.model_structure_df.copy(deep=True)
        # model_similarity_df['model'] = model_similarity_df['model'].apply(lambda x: str(x))
        model_similarity_df.set_index('model', inplace=True)

        # print(model_similarity_df.head())

        source_model_data = model_similarity_df.loc[source_model].copy(deep=True)

        struct_columns = ['struct_topology', 'struct_1', 'struct_2', 'struct_3', 'struct_4']

        for col in model_similarity_df.columns:
            model_similarity_df[col] = (model_similarity_df[col] == source_model_data[col]).astype(float)

        model_similarity_df['struct_similarity'] = 1
        source_structure_vector = model_similarity_df.loc[source_model][struct_columns].astype(float)
        for model in model_similarity_df.index:
            candidate_structure_vector = model_similarity_df.loc[model][struct_columns].astype(float)
            model_similarity_df.loc[model, 'struct_similarity'] = self.pairwise_similarity(source_structure_vector, candidate_structure_vector, sim_metric=sim_metric)

        # print(model_similarity_df.head())

        perf_similarity_df = self.model_desc.bench_df[self.model_desc.bench_df['dataset'].isin(similar_datasets)][['dataset', 'model', 'perf']].copy(deep=True)

        perf_similarity_df = perf_similarity_df.pivot(index='model', columns='dataset', values='perf')
        perf_similarity_df.reset_index(inplace=True)
        perf_similarity_df.set_index('model', inplace=True)
        perf_similarity_df.fillna(0, inplace=True)

        model_similarity_df['perf_similarity'] = 1
        for model in perf_similarity_df.index:
            model_similarity_df.loc[model, 'perf_similarity'] = self.pairwise_similarity(perf_similarity_df.loc[model], perf_similarity_df.loc[source_model], sim_metric=sim_metric)

        # print(model_similarity_df.head())

        model_similarity_df['similarity'] = (sim_weights[0] * model_similarity_df['struct_similarity'] + sim_weights[1] * model_similarity_df['perf_similarity']) / sum(sim_weights)

        model_similarity_df.sort_values(by='similarity', ascending=False, inplace=True)

        model_similarity_df.drop(index=[source_model], inplace=True)
        
        model_similarity_df = model_similarity_df.head(top_k_model)
        model_similarity_df.reset_index(inplace=True)
        model_similarity_df.drop(columns=struct_columns, inplace=True)

        model_similarity_df['has_struct_topology'] = model_similarity_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[0])
        model_similarity_df['has_struct_1'] = model_similarity_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][0])
        model_similarity_df['has_struct_2'] = model_similarity_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][1])
        model_similarity_df['has_struct_3'] = model_similarity_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][2])
        model_similarity_df['has_struct_4'] = model_similarity_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][3])
        
        return model_similarity_df


    # def get_similar_model(self, source_model_data: pd.Series, candidate_df: pd.DataFrame, topk=5, sim_metric='gower', sim_weights=[1, 1, 4]):
    #     """
    #     Retrieves the top-k similar models based on the similarity metric and weights.

    #     Parameters:
    #     - source_model_data (pd.Series): The data of the source model.
    #     - candidate_df (pd.DataFrame): The DataFrame containing the candidate models.
    #     - topk (int): The number of similar models to retrieve. Default is 5.
    #     - sim_metric (str): The similarity metric to use. Default is 'gower'.
    #     - sim_weights (list): The weights for each feature in the similarity calculation. Default is [1, 1, 4].

    #     Returns:
    #     - model_similarity_df (pd.DataFrame): The DataFrame containing the top-k similar models.
    #     """
    #     model_similarity_df = self.calculate_model_similarity(source_model_data, candidate_df, sim_metric=sim_metric, sim_weights=sim_weights)
        
    #     return model_similarity_df.head(topk)

    def calculate_complex_dataset_similarity(self, target_dataset_name, sim_metric='gower', include_target_dataset=True):
        # List of datasets
        datasets = ['cora', 'citeseer', 'pubmed', 'photo', 'computers', 'physics', 'cs', 'arxiv']

        # Initialize an empty DataFrame to store the similarity matrix
        similarity_matrix = pd.DataFrame(index=datasets, columns=datasets)

        # Compute the similarity for each pair of datasets
        for dataset1 in datasets:
            similarity_df = self.get_similar_dataset(dataset1, top_k=9, sim_metric=sim_metric, include_target_dataset=False)
            for dataset2 in datasets:
                if dataset1 == dataset2:
                    similarity_matrix.loc[dataset1, dataset2] = 1.0  # Similarity with itself is 1
                else:
                    # similarity_df = kgnas.get_similar_dataset(dataset1, top_k=9, sim_metric='gower', include_target_dataset=False)
                    similarity_value = similarity_df[similarity_df['dataset'] == dataset2]['dataset_similarity'].values[0]
                    similarity_matrix.loc[dataset1, dataset2] = similarity_value
        
        # Create an empty graph
        G = nx.DiGraph()

        # Add nodes
        for dataset in datasets:
            G.add_node(dataset)

        # Add edges with weights
        for i, dataset1 in enumerate(datasets):
            for j, dataset2 in enumerate(datasets):
                if dataset1 != dataset2:
                    weight = similarity_matrix.loc[dataset1, dataset2]
                    G.add_edge(dataset1, dataset2, weight=weight)

        # unseen_dataset = 'cora'
        personalization = {dataset: 1 if dataset == target_dataset_name else 0 for dataset in datasets}

        pagerank = nx.pagerank(G, personalization=personalization, alpha=0.9, max_iter=1000)

        return pd.DataFrame(pagerank.items(), columns=['dataset', 'dataset_similarity']).sort_values(by='dataset_similarity', ascending=False)
    
    def get_similar_dataset(self, target_dataset_name, top_k=5, sim_metric='gower', include_target_dataset=True, complex_alg=False):
        """
        Retrieves the most similar datasets to the target dataset based on a similarity metric.

        Parameters:
            target_dataset_name (str): The name of the target dataset.
            top_k (int): The number of similar datasets to retrieve (default is 5).
            sim_metric (str): The similarity metric to use (default is 'gower').
            include_target_dataset (bool): Whether to include the target dataset in the results (default is True).

        Returns:
            pandas.DataFrame: A DataFrame containing the most similar datasets.
        """
        dataset_similarities_df = self.calculate_dataset_similarity(target_dataset_name, sim_metric=sim_metric)

        if complex_alg:
            dataset_similarities_df = self.calculate_complex_dataset_similarity(target_dataset_name, sim_metric=sim_metric, include_target_dataset=include_target_dataset)

        if not include_target_dataset:
            k = min(top_k, dataset_similarities_df.shape[0] - 1)
            return dataset_similarities_df[dataset_similarities_df['dataset'] != target_dataset_name].head(k)

        return dataset_similarities_df.head(top_k)

    def recommend_model(self, target_dataset_name, top_k_dataset=5, top_k_model=5, sim_metric='l2', score_metric='avg', include_target_dataset=True, style='local'):
        """
        Recommends models based on the target dataset and their performance information.

        Parameters:
            target_dataset_name (str): The name of the target dataset.
            top_k_dataset (int): The number of similar datasets to consider.
            style (str): The recommendation style to use. Options are 'global' and 'local'.
            top_k_model (int): The number of top models to recommend for each similar dataset.
            score_metric (str): The scoring metric to use for recommendation. Options are 'avg' (average) and 'mult' (multiplication).
            include_target_dataset (bool): Flag to indicate whether to include the target dataset in the recommendations.

        Returns:
            recommend_model_df (DataFrame): A DataFrame containing the recommended models with their performance information and scores.
        """
        recommend_model_df = pd.DataFrame()

        if style == 'local':
            dataset_similarities_df = self.get_similar_dataset(target_dataset_name, top_k=top_k_dataset, sim_metric=sim_metric, include_target_dataset=include_target_dataset)

            # Obtain the top-k models for each of the similar datasets with their performance information.
            for _, row in dataset_similarities_df.iterrows():
                temp_top_model_df = self.get_top_model_from_dataset(row['dataset'], top_k=top_k_model)
                temp_top_model_df['dataset'] = row['dataset']
                temp_top_model_df['dataset_similarity'] = row['dataset_similarity']
                temp_top_model_df['standardized_perf'] = (temp_top_model_df['perf'] - temp_top_model_df['perf'].min()) / (temp_top_model_df['perf'].max() - temp_top_model_df['perf'].min())
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
            
            # Standardize the models based on their interial performance with the dataset.
            perf = recommend_model_df['standardized_perf'].to_numpy()
            
            if score_metric == 'avg':
                recommend_model_df['score'] = (dataset_similarity + perf) / 2
            if score_metric == 'mult':
                recommend_model_df['score'] = dataset_similarity * perf
        
        if style == 'global':
            dataset_similarities_df = self.get_similar_dataset(target_dataset_name, top_k=top_k_dataset, sim_metric=sim_metric, include_target_dataset=include_target_dataset)
            
            similar_datasets = dataset_similarities_df['dataset'].to_list()
            # print(dataset_similarities_df)
            
            related_models = self.model_desc.hyper_relation_info.loc[self.model_desc.hyper_relation_info['source_entity'].isin(similar_datasets)].copy(deep=True)

            related_models['dataset_similarity'] = related_models['source_entity'].apply(lambda x: dataset_similarities_df[dataset_similarities_df['dataset'] == x]['dataset_similarity'].values[0])
            related_models['weighted_score'] = related_models['perf_score'] * related_models['dataset_similarity']

            related_models['norm_target_entity'] = related_models['target_entity'].apply(lambda x: x[1:])

            recommend_model_df = related_models.groupby('norm_target_entity').agg({'weighted_score': 'mean'}).reset_index()

            recommend_model_df = recommend_model_df.sort_values(by='weighted_score', ascending=False).head(top_k_model)

            recommend_model_df = recommend_model_df.rename(columns={'norm_target_entity': 'model'})
            recommend_model_df['model'] = recommend_model_df['model'].apply(lambda x: int(x))

            recommend_model_df['has_struct_topology'] = recommend_model_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[0])
            recommend_model_df['has_struct_1'] = recommend_model_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][0])
            recommend_model_df['has_struct_2'] = recommend_model_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][1])
            recommend_model_df['has_struct_3'] = recommend_model_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][2])
            recommend_model_df['has_struct_4'] = recommend_model_df['model'].apply(lambda x: self.model_desc.decoder.decode_hash(x, False)[1][3])
            # lk, op = self.model_desc.decoder.decode_hash(hash_value, use_proteins)

        return recommend_model_df
        
    def get_top_model_from_dataset(self, dataset_name, perf_metric='perf', top_k=10):
        """
        Retrieves the top models from a given dataset based on a performance metric.

        Parameters:
        - dataset_name (str): The name of the dataset.
        - perf_metric (str): The performance metric to sort the models by. Default is 'perf'.
        - top_k (int): The number of top models to retrieve. Default is 10.

        Returns:
        - top_model_df (DataFrame): A DataFrame containing the top models and their corresponding performance metric.
        """
        top_model_df = self.model_desc.hyper_relation_info[self.model_desc.hyper_relation_info['source_entity'] == dataset_name].sort_values(by=perf_metric, ascending=False).head(top_k).copy(deep=True)
        top_model_df = top_model_df.drop(columns=['source_entity'])

        top_model_df.sort_values(by=perf_metric, ascending=False, inplace=True)
        top_model_df.rename(columns={'target_entity': 'model'}, inplace=True)

        return top_model_df

    def visualize(self):
        pass

    def set_num_weight(self, weight):
        self.numerical_weight = weight
        self.categorical_weight = 1 - weight
    
    def generate_graph(self):
        model_2ary_edges = [(row['source_entity'], row['target_entity'], {'relation': row['relation']}) for idx, row in self.model_desc.two_ary_info.iterrows()]
        self.KG.add_edges_from(model_2ary_edges)

        data_2ary_edges = [(row['source_entity'], row['target_entity'], {'relation': row['relation']}) for idx, row in self.dataset_desc.two_ary_info.iterrows()]
        self.KG.add_edges_from(data_2ary_edges)
        
        hyper_relation_names = list(self.model_desc.hyper_relation_info.columns.difference(['source_entity', 'target_entity']))
        model_hyper_edges = [(row['source_entity'], row['target_entity'], {key: row[key] for key in hyper_relation_names}) for idx, row in self.model_desc.hyper_relation_info.iterrows()]
        self.KG.add_edges_from(model_hyper_edges)

        self.entities = set(self.KG.nodes)

    def save_graph(self, dir='./KG/'):
        if not os.path.exists(dir):
            os.makedirs(dir)

        data = nx.node_link_data(self.KG)
        with open(dir+"KGNAS_graph.json", 'w') as f:
            json.dump(data, f)

    def load_graph(self, dir='./KG/'):
        with open(dir+"KGNAS_graph.json", 'r') as f:
            data = json.load(f)
            self.KG = nx.node_link_graph(data)

    def save_knowledge_graph(self):
        self.dataset_desc.generate_knowledge_graph()
        self.model_desc.generate_knowledge_graph()

        entities = pd.concat([self.dataset_desc.entities, self.model_desc.entities]).drop_duplicates(subset=['name', 'macro_type', 'micro_type'])
        relations = pd.concat([self.dataset_desc.relations, self.model_desc.relations])

        entities['id'] = range(len(entities))
        relations['id'] = range(len(relations))

        entity_name_to_id = {name: id for name, id in zip(entities['name'], entities['id'])}

        relations['target_entity'] = relations['target_entity'].apply(lambda x: entity_name_to_id[x])
        relations['source_entity'] = relations['source_entity'].apply(lambda x: entity_name_to_id[x])

        print('num macro types', len(entities['macro_type'].unique()))
        print('num micro types', len(entities['micro_type'].unique()))
        print('num entities', len(entities))

        print('num relation types', len(relations['relation'].unique()))
        print('num relations', len(relations))

        # for row in self.dataset_desc.uniary_info.iterrows():
        #     row_dict = {'name': row['dataset'], 'macro_type': 'dataset', 'micro_type': 'dataset', 'property': row.drop('dataset').to_dict()}
        #     entities.append(row_dict)

        # for row in self.model_desc.uniary_info.iterrows():
        #     row_dict = {'name': row['model'], 'macro_type': 'model', 'micro_type': 'model', 'property': row.drop('model').to_dict()}
        #     entities.append(row_dict)

        # for row in self.dataset_desc.two_ary_info.iterrows():
        #     row_dict = {'name': row['target_entity'], 'macro_type': 'dataset', 'micro_type': row['relation'][4:], 'property': None}
        
        # for row in self.model_desc.two_ary_info.iterrows():
        #     row_dict = {'name': row['target_entity'], 'macro_type': 'model', 'micro_type': row['relation'][4:], 'property': None}

        # for row in self.dataset_desc.two_ary_info.iterrows():
        #     row_dict = {'source_entity': row['source_entity'], 'target_entity': row['target_entity'], 'type': 'dataset_related', 'property': {'relation': row['relation']}}
        #     relations.append(row_dict)

        # for row in self.model_desc.two_ary_info.iterrows():
        #     row_dict = {'source_entity': row['source_entity'], 'target_entity': row['target_entity'], 'type': 'model_related', 'property': {'relation': row['relation']}}
        #     relations.append(row_dict)

        # for row in self.model_desc.hyper_relation_info.iterrows():
        #     row_dict = {'source_entity': row['source_entity'], 'target_entity': row['target_entity'], 'type': 'model_dataset', 'property': row.drop(['source_entity', 'target_entity']).to_dict()}
        #     relations.append(row_dict)

        # knowledge_graph = {'entities': entities, 'relations': relations}

        with open(self.kg_dir+"KGNAS_knowledge_graph_entities.json", 'w') as f:
            json.dump(entities.to_dict('records'), f)

        with open(self.kg_dir+"KGNAS_knowledge_graph_relations.json", 'w') as f:
            json.dump(relations.to_dict('records'), f)

    def load_knowledge_graph(self):
        with open(self.kg_dir+"KGNAS_knowledge_graph_entities.json", 'r') as f:
            entities = json.load(f)
        
        with open(self.kg_dir+"KGNAS_knowledge_graph_relations.json", 'r') as f:
            relations = json.load(f)

        entity_id_to_name = {id: name for id, name in zip([entity['id'] for entity in entities], [entity['name'] for entity in entities])}

        # entities = knowledge_graph['entities']
        # relations = knowledge_graph['relations']

        dataset_entities = [entity for entity in entities if entity['macro_type'] == 'dataset']
        dataset_entity_df = {}
        dataset_entity_df['dataset'] = [entity['name'] for entity in dataset_entities]
        property_names = dataset_entities[0]['property'].keys()
        for property in property_names:
            dataset_entity_df[property] = [entity['property'][property] for entity in dataset_entities]
        self.dataset_desc.uniary_info = pd.DataFrame(dataset_entity_df)

        model_entities = [entity for entity in entities if entity['macro_type'] == 'model']
        model_entity_df = {}
        model_entity_df['model'] = [entity['name'] for entity in model_entities]
        property_names = model_entities[0]['property'].keys()
        for property in property_names:
            model_entity_df[property] = [entity['property'][property] for entity in model_entities]
        self.model_desc.uniary_info = pd.DataFrame(model_entity_df)

        dataset_binary_relations = [relation for relation in relations if relation['macro_type'] == 'dataset']
        dataset_binary_relation_df = {}
        dataset_binary_relation_df['source_entity'] = [entity_id_to_name[relation['source_entity']] for relation in dataset_binary_relations]
        dataset_binary_relation_df['target_entity'] = [entity_id_to_name[relation['target_entity']] for relation in dataset_binary_relations]
        dataset_binary_relation_df['relation'] = [relation['relation'] for relation in dataset_binary_relations]
        self.dataset_desc.two_ary_info = pd.DataFrame(dataset_binary_relation_df)

        model_binary_relations = [relation for relation in relations if relation['macro_type'] == 'model' and relation['property'] == None]
        model_binary_relation_df = {}
        model_binary_relation_df['source_entity'] = [entity_id_to_name[relation['source_entity']] for relation in model_binary_relations]
        model_binary_relation_df['target_entity'] = [entity_id_to_name[relation['target_entity']] for relation in model_binary_relations]
        model_binary_relation_df['relation'] = [relation['relation'] for relation in model_binary_relations]
        self.model_desc.two_ary_info = pd.DataFrame(model_binary_relation_df)

        model_hyper_relations = [relation for relation in relations if relation['macro_type'] == 'model' and relation['property'] != None]
        model_hyper_relation_df = {}
        model_hyper_relation_df['source_entity'] = [entity_id_to_name[relation['source_entity']] for relation in model_hyper_relations]
        model_hyper_relation_df['target_entity'] = [entity_id_to_name[relation['target_entity']] for relation in model_hyper_relations]
        property_names = model_hyper_relations[0]['property'].keys()
        for property in property_names:
            model_hyper_relation_df[property] = [relation['property'][property] for relation in model_hyper_relations]
        self.model_desc.hyper_relation_info = pd.DataFrame(model_hyper_relation_df)

    def get_knowledge_graph(self):
        return self.KG

    def summrize_knowledge_graph(self, generate=False):
        if not os.path.exists(self.kg_dir+"KGNAS_graph.json") or generate:
            self.generate_graph()
        else:
            self.load_graph(self.kg_dir)
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

        num_binary_relations = len(self.model_desc.two_ary_info) + len(self.dataset_desc.two_ary_info)
        num_hyper_relations = len(self.model_desc.hyper_relation_info)

        print(f"Number of binary relations: {num_binary_relations}")
        print(f"Number of hyper relations: {num_hyper_relations}")

        num_models = len(self.model_desc.hyper_relation_info['target_entity'].unique())
        num_datasets = len(self.dataset_desc.two_ary_info['source_entity'].unique())

        print(f"Number of models: {num_models}")
        print(f"Number of datasets: {num_datasets}")

    def add_dataset_description(self, dataset_name, dataset=None, semantic_description=None, root_dir='datasets/', num_samples=20, num_hops=2, seed=42):
        self.dataset_desc.add_description(dataset_name, dataset=dataset, semantic_description=semantic_description, root_dir=root_dir, num_samples=num_samples, num_hops=num_hops, seed=seed)

    def process_numerical_data(self, column, process_method='normal', target_index=None):
        column = column.fillna(0)

        if process_method == 'normal':
            column = (column - column.min()) / (column.max() - column.min())
            return column, None
        
        if process_method == 'outlier':
            exclude_column = column[column.index != target_index]
            max_val = exclude_column.max()
            min_val = exclude_column.min()
            exclude_column = (exclude_column - min_val) / (max_val - min_val)
            Q1 = exclude_column.quantile(0.25)
            Q3 = exclude_column.quantile(0.75)
            IQR = Q3 - Q1
            column = (column - min_val) / (max_val - min_val)
            
            return column, IQR
    
    def pairwise_similarity(self, target_vector, bench_vector, sim_metric='gower', numerical_columns=None, categorical_columns=None, standardize=False, activation='abs', dynamic_upperbound=None, return_raw=False):
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
                        
            if standardize:
                target_num_vector = (target_num_vector - target_num_vector.mean()) / target_num_vector.std()
                bench_num_vector = (bench_num_vector - bench_num_vector.mean()) / bench_num_vector.std()

            numerical_similarity_vector = np.abs(target_num_vector - bench_num_vector)

            if activation == 'power':
                if self.power < 0:
                    numerical_similarity_vector = 1 - np.power(1 - numerical_similarity_vector, -self.power)
                else:
                    numerical_similarity_vector = np.power(numerical_similarity_vector, self.power)
            elif activation == 'tanh':
                numerical_similarity_vector = np.tanh(numerical_similarity_vector)
            

            bound = dynamic_upperbound if dynamic_upperbound is not None else self.upper_bound

            # print(bound)

            if self.power < 0 and activation == 'power':
                numerical_similarity_vector = numerical_similarity_vector * bound + self.lower_bound
            else:
                numerical_similarity_vector = np.minimum(numerical_similarity_vector, bound) + self.lower_bound

            if return_raw:
                return 1 - numerical_similarity_vector
            
            numerical_similarity = np.mean(1 - numerical_similarity_vector)

            categorical_similarity = dice_similarity(target_cat_vector, bench_cat_vector)
            
            similarity = self.numerical_weight * numerical_similarity + self.categorical_weight * categorical_similarity
            
            return similarity

        if numerical_columns is not None:
            target_vector = target_vector[numerical_columns].to_numpy()
            bench_vector = bench_vector[numerical_columns].to_numpy()
        else:
            target_vector = target_vector.to_numpy()
            bench_vector = bench_vector.to_numpy()

        target_vector = target_vector.astype(np.float32)
        bench_vector = bench_vector.astype(np.float32)
        # print(target_vector, bench_vector)

        if sim_metric == 'cosine':
            # if numerical_columns == None:
            return torch.nn.functional.cosine_similarity(torch.tensor(target_vector), torch.tensor(bench_vector), dim=0).item()
        if sim_metric == 'l2':
            # if numerical_columns == None:
            return 1 - torch.nn.functional.mse_loss(torch.tensor(target_vector), torch.tensor(bench_vector)).item()
            # else:
            #     return 1 - torch.nn.functional.mse_loss(torch.tensor(target_vector[numerical_columns]), torch.tensor(bench_vector[numerical_columns]), p=2).item()
        if sim_metric == 'l1':
            # if numerical_columns == None:
            return 1 - torch.nn.functional.l1_loss(torch.tensor(target_vector), torch.tensor(bench_vector)).item()
            # else:
            #     return 1 - torch.nn.functional.l1_loss(torch.tensor(target_vector[numerical_columns]), torch.tensor(bench_vector[numerical_columns])).item()
            # return 1 - torch.nn.functional.l1_loss(torch.tensor(target_vector), torch.tensor(bench_vector)).item()
        
    def get_similarity_vector(self, source_dataset, target_dataset):
        source_vector = self.normalized_data_desc_df.loc[source_dataset]
        target_vector = self.normalized_data_desc_df.loc[target_dataset]

        similarity_vector = self.pairwise_similarity(source_vector, target_vector, sim_metric='gower', numerical_columns=self.dataset_numerical_columns, categorical_columns=self.dataset_categorical_columns, standardize=self.standardize, activation=self.activation, dynamic_upperbound=self.dynamic_upperbound_dict[source_dataset], return_raw=True)

        return similarity_vector