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

# structure_description = {}
# hyper_parameter_description = {}

class ModelDescription:
    def __init__(self, kgdir='./KG/', load=False):
        self.datasets = []
        self.uniary_info = None
        self.two_ary_info = None
        self.hyper_relation_info = None
        self.kgdir = kgdir

        self.hardware_df = pd.DataFrame(HARDWARE)
        self.hyper_param_df = pd.DataFrame(HYPER_PARAM)
        self.data_related_df = pd.concat([self.hardware_df, self.hyper_param_df.drop('dataset', axis=1)], axis=1)

        self.bench_df = self._read_bench_data()
        self.model_structure_df = self._decompose_model()

        self.relation_names = None

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
        self.relation_names['performance'] = [col for col in self.bench_df.columns if col not in ['dataset', 'model']]
        self.relation_names['structure'] = [col for col in self.model_structure_df.columns if col not in ['dataset', 'model']]

        data_model_df = self.bench_df.copy(deep=True)
        data_model_df = pd.merge(data_model_df, self.model_structure_df, on='model', how='left')
        data_model_df = pd.merge(data_model_df, self.data_related_df, on='dataset', how='left')

        all_column_names = list(data_model_df.columns)
        performance_columns = ['valid_perf', 'perf', 'latency', 'para']
        non_performance_columns = [col for col in all_column_names if col not in performance_columns + ['dataset', 'model']]

        dataset_to_index = {dname: i for i, dname in enumerate(data_model_df['dataset'].unique())}
        data_model_df['indicator'] = data_model_df['dataset'].apply(lambda x: str(dataset_to_index[x]))
        data_model_df['model'] = data_model_df['indicator'] +  data_model_df['model'].apply(lambda x: str(x))
        data_model_df.drop('indicator', axis=1, inplace=True)

        # Uniary information
        self.uniary_info = data_model_df[non_performance_columns].select_dtypes(include=['number']).copy(deep=True)
        self.uniary_info = pd.concat([data_model_df['model'], self.uniary_info], axis=1)

        # Two-ary information
        temp_two_ary_info = data_model_df[non_performance_columns].select_dtypes(include=['object']).copy(deep=True)
        self.two_ary_info = pd.DataFrame()
        for col in temp_two_ary_info.columns:
            relation = 'has_' + col.replace(' ', '_')
            for relation_type in self.relation_names.keys():
                if col in self.relation_names[relation_type]:
                    self.relation_names[relation_type].remove(col)
                    self.relation_names[relation_type].append(relation)

            temp_df = pd.concat([data_model_df['model'], temp_two_ary_info[col]], axis=1)
            temp_df['relation'] = relation
            temp_df.rename(columns={col: 'target_entity', 'model': 'source_entity'}, inplace=True)
            self.two_ary_info = pd.concat([self.two_ary_info, temp_df])

        # Hyper relation information
        self.hyper_relation_info = data_model_df[['dataset', 'model'] + performance_columns].copy(deep=True)
        self.hyper_relation_info.rename(columns={'dataset': 'source_entity', 'model': 'target_entity'}, inplace=True)
        self.hyper_relation_info['relation'] = 'has_performance'

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

        decoder = HashDecoder(GNN_LIST, GNN_LIST_PROTEINS)

        # print('non protein model code:', non_protein_model_df['model'].unique()[:10])

        non_protein_model_df['struct_topology'] = non_protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, False)[0])
        non_protein_model_df['struct_1'] = non_protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, False)[1][0])
        non_protein_model_df['struct_2'] = non_protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, False)[1][1])
        non_protein_model_df['struct_3'] = non_protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, False)[1][2])
        non_protein_model_df['struct_4'] = non_protein_model_df['model'].apply(lambda x: decoder.decode_hash(x, False)[1][3])

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