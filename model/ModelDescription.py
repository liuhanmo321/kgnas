from nas_bench_graph import light_read
import pandas as pd
import numpy as np
import json
from .HashDecoder import HashDecoder

GNN_LIST = [
    "gat",  # GAT with 2 heads
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  # k-GNN
    "fc",  # fully-connected
    "skip"  # skip connection
]

GNN_LIST_PROTEINS = [
    "gcn",  # GCN
    "sage",  # sage
    "arma",
    "fc",  # fully-connected
    "skip"  # skip connection
]

HYPER_PARAM = {
    "dataset": ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins'],
    "NumPrevLayer": [0, 0, 0, 1, 1, 1, 1, 0, 1],
    "NumPostLayer": [1, 1, 0, 0, 1, 0, 1, 1, 1],
    "Dimension": [256, 256, 128, 128, 256, 128, 64, 128, 256],
    "Dropout": [0.7, 0.7, 0.3, 0.6, 0.4, 0.7, 0.1, 0.2, 0.0],
    "Optimizer": ["SGD", "SGD", "SGD", "SGD", "SGD", "Adam", "Adam", "Adam", "Adam"],
    "LR": [0.1, 0.2, 0.2, 0.5, 0.01, 0.0002, 0.005, 0.002, 0.01],
    "WD": [0.0005, 0.0005, 0.0005, 0.0005, 0.0, 0.0005, 0.0005, 0.0, 0.0005],
    "NumEpoch": [400, 400, 500, 400, 200, 500, 500, 500, 500],
}

HARDWARE = {
    "dataset": ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins'],
    "CPU": [
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # Cora
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # CiteSeer
        "Intel(R) Xeon(R) Gold 6129 CPU @ 2.30GHz",  # PubMed
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # Coauthor-CS
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # Coauthor-Physics
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # Amazon-Photo
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz",  # Amazon-Computers
        "Intel(R) Xeon(R) Gold 6129 CPU @ 2.30GHz",  # ogbn-arXiv
        "Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz"   # ogbn-proteins
        ],
    "GPU": [
        "NVIDIA Tesla V100 with 16GB",  # Cora
        "NVIDIA Tesla V100 with 16GB",  # CiteSeer
        "NVIDIA GeForce RTX 3090 with 24GB",  # PubMed
        "NVIDIA Tesla V100 with 16GB",  # Coauthor-CS
        "NVIDIA Tesla V100 with 16GB",  # Coauthor-Physics
        "NVIDIA Tesla V100 with 16GB",  # Amazon-Photo
        "NVIDIA Tesla V100 with 16GB",  # Amazon-Computers
        "NVIDIA GeForce RTX 3090 with 24GB",  # ogbn-arXiv
        "NVIDIA Tesla V100 with 16GB"   # ogbn-proteins
        ],
    "OperatingSystem": [
        "CentOS Linux release 7.6.1810",  # Cora
        "CentOS Linux release 7.6.1810",  # CiteSeer
        "Ubuntu 18.04.6 LTS",             # PubMed
        "CentOS Linux release 7.6.1810",  # Coauthor-CS
        "CentOS Linux release 7.6.1810",  # Coauthor-Physics
        "CentOS Linux release 7.6.1810",  # Amazon-Photo
        "CentOS Linux release 7.6.1810",  # Amazon-Computers
        "Ubuntu 18.04.6 LTS",             # ogbn-arXiv
        "CentOS Linux release 7.6.1810"   # ogbn-proteins
        ]
}

SENANTICS = {
    "dataset": ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins'],
    "task": ['Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification'], 
    "modality": ['Graph', 'Graph', 'Graph', 'Graph', 'Graph', 'Graph', 'Graph', 'Graph', 'Graph'],
}

# structure_description = {}
# hyper_parameter_description = {}

class ModelDescription:
    def __init__(self, kg_dir='./KG/', load=False):
        self.datasets = []
        self.uniary_info = None
        self.two_ary_info = None
        self.hyper_relation_info = None
        self.kg_dir = kg_dir

        self.decoder = HashDecoder(GNN_LIST, GNN_LIST_PROTEINS)

        self.hardware_df = pd.DataFrame(HARDWARE)
        self.hyper_param_df = pd.DataFrame(HYPER_PARAM)
        self.semantic_df = pd.DataFrame(SENANTICS)
        self.data_related_df = pd.concat([self.hardware_df, self.hyper_param_df.drop('dataset', axis=1), self.semantic_df.drop('dataset', axis=1)], axis=1)

        self.bench_df = self._read_bench_data()
        self.model_structure_df = self._decompose_model()

        self.relation_names = None

        self.entities = None
        self.relations = None
        
        if load:
            with open(self.kg_dir+"KGNAS_knowledge_graph.json", 'r') as f:
                knowledge_graph = json.load(f)

            entities = knowledge_graph['entities']
            relations = knowledge_graph['relations']

            model_entities = [entity for entity in entities if entity['type'] == 'model']
            model_entity_df = {}
            model_entity_df['model'] = [entity['name'] for entity in model_entities]
            property_names = model_entities[0]['property'].keys()
            for property in property_names:
                model_entity_df[property] = [entity['property'][property] for entity in model_entities]
            self.uniary_info = pd.DataFrame(model_entity_df)

            model_binary_relations = [relation for relation in relations if relation['type'] == 'model_related']
            model_binary_relation_df = {}
            model_binary_relation_df['source_entity'] = [relation['source_entity'] for relation in model_binary_relations]
            model_binary_relation_df['target_entity'] = [relation['target_entity'] for relation in model_binary_relations]
            model_binary_relation_df['relation'] = [relation['property']['relation'] for relation in model_binary_relations]
            self.two_ary_info = pd.DataFrame(model_binary_relation_df)

            model_hyper_relations = [relation for relation in relations if relation['type'] == 'model_dataset']
            model_hyper_relation_df = {}
            model_hyper_relation_df['source_entity'] = [relation['source_entity'] for relation in model_hyper_relations]
            model_hyper_relation_df['target_entity'] = [relation['target_entity'] for relation in model_hyper_relations]
            property_names = model_hyper_relations[0]['property'].keys()
            for property in property_names:
                model_hyper_relation_df[property] = [relation['property'][property] for relation in model_hyper_relations]
            self.hyper_relation_info = pd.DataFrame(model_hyper_relation_df)
        
        self.prepare_knowledge_graph()

    
    # def get_descriptions(self):
    #     return self.descriptions
    
    # def add_description(self, dataset, dataset_name):
    #     pass

    def prepare_knowledge_graph(self):
        # non_performance_columns = self.bench_df.columns.difference(performance_columns)
        self.relation_names = {}
        self.relation_names['hyper_param'] = [key for key in HYPER_PARAM.keys() if key != 'dataset']
        self.relation_names['hardware'] = [key for key in HARDWARE.keys() if key != 'dataset']
        self.relation_names['semantics'] = [key for key in SENANTICS.keys() if key != 'dataset']
        self.relation_names['performance'] = [col for col in self.bench_df.columns if col not in ['dataset', 'model']]
        self.relation_names['structure'] = [col for col in self.model_structure_df.columns if col not in ['dataset', 'model']]

        data_model_df = self.bench_df.copy(deep=True)
        data_model_df = pd.merge(data_model_df, self.model_structure_df, on='model', how='left')
        data_model_df = pd.merge(data_model_df, self.data_related_df, on='dataset', how='left')

        all_column_names = list(data_model_df.columns)
        performance_columns = ['valid_perf', 'perf', 'latency', 'para']
        non_performance_columns = [col for col in all_column_names if col not in performance_columns + ['dataset', 'model']]
        # print(non_performance_columns)

        dataset_to_index = {dname: i for i, dname in enumerate(data_model_df['dataset'].unique())}
        data_model_df['indicator'] = data_model_df['dataset'].apply(lambda x: str(dataset_to_index[x]))
        data_model_df['model'] = data_model_df['indicator'] +  data_model_df['model'].apply(lambda x: str(x))
        data_model_df.drop('indicator', axis=1, inplace=True)

        # Uniary information
        # uniary_special_columns = ['model']
        self.uniary_info = data_model_df[non_performance_columns].select_dtypes(include=['number']).copy(deep=True)
        self.uniary_info = pd.concat([data_model_df['model'], self.uniary_info], axis=1)

        # Two-ary information
        temp_two_ary_info = data_model_df[non_performance_columns].select_dtypes(include=['object']).copy(deep=True)
        self.two_ary_info = pd.DataFrame()
        for col in temp_two_ary_info.columns:
            relation = 'has_' + col.replace(' ', '_')
            for relation_type in self.relation_names.keys():
                if col in self.relation_names[relation_type]:
                    self.relation_names[relation_type].append(relation)
                    self.relation_names[relation_type].remove(col)

            temp_df = pd.concat([data_model_df['model'], temp_two_ary_info[col]], axis=1)
            temp_df['relation'] = relation
            temp_df.rename(columns={col: 'target_entity', 'model': 'source_entity'}, inplace=True)
            self.two_ary_info = pd.concat([self.two_ary_info, temp_df])

        for relation_type in self.relation_names.keys():
            # Create a new list to store relations that should be kept
            new_relations = []
            for relation in self.relation_names[relation_type]:
                if relation in list(self.two_ary_info['relation'].unique()):
                    new_relations.append(relation)
            # Replace the original list with the new list
            self.relation_names[relation_type] = new_relations
            
        # Hyper relation information
        self.hyper_relation_info = data_model_df[['dataset', 'model'] + performance_columns].copy(deep=True)
        for dataset in self.hyper_relation_info['dataset'].unique():
            for col in ['perf', 'valid_perf', 'latency', 'para']:
                self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_rank'] = self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col].rank(ascending=True, method='average')
                self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_rank'] = self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_rank'] / self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col].count()
                self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_top_diff'] = 1 - (self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col].max() - self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col]) / self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col].max()
            
                self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_score'] = (self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col] + self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col+'_rank'] + self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col+'_top_diff']) / 3
                # self.hyper_relation_info.loc[self.hyper_relation_info['dataset'] == dataset, col+'_mid_diff'] = self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col].median() - self.hyper_relation_info[self.hyper_relation_info['dataset'] == dataset][col]
        self.hyper_relation_info.rename(columns={'dataset': 'source_entity', 'model': 'target_entity'}, inplace=True)
        self.hyper_relation_info['relation'] = 'has_performance'
        self.hyper_relation_info['task'] = 'Node Classification'
        self.hyper_relation_info['structure_id'] = self.hyper_relation_info['target_entity'].apply(lambda x: x[1:])

    def generate_knowledge_graph(self):
        self.entities = self.uniary_info.copy(deep=True)
        self.entities['macro_type'] = 'model'
        self.entities['micro_type'] = 'model'
        self.entities['property'] = [row.drop('model').to_dict() for idx, row in self.uniary_info.iterrows()]
        # self.entities['id'] = ['model_model_' + str(i) for i in range(len(self.uniary_info))]
        self.entities.rename(columns={'model': 'name'}, inplace=True)
        self.entities = self.entities[['name', 'macro_type', 'micro_type', 'property']]
        
        temp_entities = self.two_ary_info.copy(deep=True)
        temp_entities['macro_type'] = ['model' if row['relation'] not in ['has_task', 'has_modality'] else row['relation'][4:] for idx, row in temp_entities.iterrows()]
        temp_entities['micro_type'] = temp_entities['relation'].apply(lambda x: x[4:])
        temp_entities['property'] = None
        # temp_entities['id'] = ['model_' + row['micro_type'] + str(idx) for idx, row in temp_entities.iterrows() if row['micro_type'] not in ['task', 'modality']]
        temp_entities.rename(columns={'target_entity': 'name'}, inplace=True)
        temp_entities = temp_entities[['name', 'macro_type', 'micro_type', 'property']]

        self.entities = pd.concat([self.entities, temp_entities])

        self.relations = self.two_ary_info.copy(deep=True)

        # self.relations['target_entity'] = self.relations['target_entity'].apply(lambda x: self.entities[self.entities['name'] == x]['id'].values[0])
        # self.relations['source_entity'] = self.relations['source_entity'].apply(lambda x: self.entities[self.entities['name'] == x]['id'].values[0])
        # self.relations['id'] = ['model_' + row['relation'] + str(idx) for idx, row in self.relations.iterrows()]
        self.relations['property'] = None
        self.relations['macro_type'] = 'model'

        self.relations = self.relations[['source_entity', 'target_entity', 'macro_type', 'relation', 'property']]

        temp_relations = self.hyper_relation_info.copy(deep=True)
        # temp_relations['source_entity'] = temp_relations['source_entity'].apply(lambda x: self.entities[self.entities['name'] == x]['id'].values[0])
        # temp_relations['target_entity'] = temp_relations['target_entity'].apply(lambda x: self.entities[self.entities['name'] == x]['id'].values[0])
        # temp_relations['id'] = ['model_performance_' + str(idx) for idx, row in temp_relations.iterrows()]
        temp_relations['property'] = [row.drop(['source_entity', 'target_entity', 'relation']).to_dict() for idx, row in temp_relations.iterrows()]
        temp_relations['macro_type'] = 'model'

        temp_relations = temp_relations[['source_entity', 'target_entity', 'macro_type', 'relation', 'property']]

        self.relations = pd.concat([self.relations, temp_relations])

    def _read_bench_data(self):
        dnames = ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins']

        all_bench_df = pd.DataFrame()
        for dname in dnames:
            # read the benchmark
            bench = light_read(dname) 
            bench_df = pd.DataFrame.from_dict(bench, orient='index')
            
            # filter the models, they are stored in codes, the correspondance between the concrete structure and code will be processed later.
            bench_df.reset_index(inplace=True)
            bench_df.rename(columns={'index': 'model'}, inplace=True)
            
            # create the column of data sets
            bench_df['dataset'] = dname

            # combine the data into a giant dataframe
            all_bench_df = pd.concat([all_bench_df, bench_df])
            # print(dname, bench_df.shape)
            # print(all_bench_df.shape)

        all_bench_df = all_bench_df[['dataset', 'model', 'valid_perf', 'perf', 'latency', 'para']]

        # for name in ['valid_perf', 'perf', 'latency', 'para']:
        #     all_bench_df.rename(columns={name: f'perf_{name}'}, inplace=True)
        
        return all_bench_df
    
    def _decompose_model(self):
        non_protein_model_df = self.bench_df[self.bench_df['dataset'] == 'cora'][['dataset', 'model']].copy(deep=True)
        # protein_model_df = self.bench_df[self.bench_df['dataset'] == 'proteins'][['dataset', 'model']].copy(deep=True)

        # model_df = self.bench_df[self.bench_df['dataset'] == 'cora'][['dataset', 'model']].copy(deep=True)

        # decoder = HashDecoder(GNN_LIST, GNN_LIST_PROTEINS)

        # print('non protein model code:', non_protein_model_df['model'].unique()[:10])

        non_protein_model_df['struct_topology'] = non_protein_model_df['model'].apply(lambda x: self.decoder.decode_hash(x, False)[0])
        non_protein_model_df['struct_1'] = non_protein_model_df['model'].apply(lambda x: self.decoder.decode_hash(x, False)[1][0])
        non_protein_model_df['struct_2'] = non_protein_model_df['model'].apply(lambda x: self.decoder.decode_hash(x, False)[1][1])
        non_protein_model_df['struct_3'] = non_protein_model_df['model'].apply(lambda x: self.decoder.decode_hash(x, False)[1][2])
        non_protein_model_df['struct_4'] = non_protein_model_df['model'].apply(lambda x: self.decoder.decode_hash(x, False)[1][3])

        # print('protein model code:', protein_model_df['model'].unique()[:10])

        # protein_model_df['struct_topology'] = protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, True)[0])
        # protein_model_df['struct_1'] = protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, True)[1][0])
        # protein_model_df['struct_2'] = protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, True)[1][1])
        # protein_model_df['struct_3'] = protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, True)[1][2])
        # protein_model_df['struct_4'] = protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, True)[1][3])

        # model_df = pd.concat([non_protein_model_df, protein_model_df])
        model_df = non_protein_model_df
        model_df['struct_topology'] = model_df['struct_topology'].apply(lambda x: str(x))

        model_df.drop('dataset', axis=1, inplace=True)

        return model_df