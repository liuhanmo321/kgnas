from .GraphDatasetUnderstanding import GraphDatasetUnderstanding
import pandas as pd
import json

EXISTING_DESCRIPTION = {
    # 'type': ['Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous', 'Homogeneous'], 
    'node_feature': [1433, 3703, 500, 6805, 8415, 745, 767, 128, 0], 
    # 'assortativity': [-0.06587087427227857, 0.048378078374214546, -0.04364031570334561, 0.11260854329931977, 0.2010313810362111, -0.04494270527930541, -0.056487705164197634, -0.04311895177607261, 0.35520272226167543], 
    # 'feature_type': ['Node', 'Node', 'Node', 'Node', 'Node', 'Node', 'Node', 'Node', 'Edge'], 
    'local_average_clustering_coefficient': [None, None, None, None, None, None, None, None, 0.6573941635441265], 
    'average_clustering_coefficient': [0.24067329850193728, 0.14147102442629086, 0.060175209437523615, 0.34252338196985804, 0.3776229390845259, 0.4039792743937545, 0.34412638745332397, 0.22612914062470202, None], 
    'average_degree_centrality': [0.0014399999126942077, 0.0008227297529768376, 0.00022803908825811387, 0.0004873474441645473, 0.0004168365381774087, 0.004070112116838717, 0.0026002762758509414, 8.074789913670332e-05, 0.004504521754591681], 
    'local_average_betweenness_centrality': [0.0, 0.06637426900584795, 0.04038367546432063, 0.034573103206115105, 0.01289624003515159, 0.002738664262816096, 0.003674253329531406, 0.020211893569974666, 0.0051445249418492715], 
    'edge_count': [5278, 4552, 44324, 81894, 247962, 119081, 245861, 1157799, 39561252], 
    'num_classes': [7, 6, 3, 15, 5, 8, 10, 40, 112], 
    'density': [0.0014399999126942077, 0.0008227297529768376, 0.00022803908825811382, 0.00048734744416454727, 0.0004168365381774087, 0.004070112116838717, 0.0026002762758509414, 8.074789913670332e-05, 0.004504521754591681], 
    'average_eigenvector_centrality': [0.0047865456098027765, 0.0033854307488708703, 0.0014101036104415428, 0.0023585311444032655, 0.0016277767776128051, 0.004240506438932563, 0.0033683404991575005, 0.0007299364762486809, 0.0005329522847746617], 
    # 'task': ['Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification', 'Node Classification'], 
    'local_average_closeness_centrality': [1.0, 0.4697435587367421, 0.4524748732421055, 0.4244744169883969, 0.30603317639876243, 0.3746486414543327, 0.3940379176382557, 0.3338630446056274, 0.5694732422316185], 
    'node_count': [2708, 3327, 19717, 18333, 34493, 7650, 13752, 169343, 132534], 
    'average_degree': [3.8980797636632203, 2.7363991584009617, 4.496018664096972, 8.93405334642448, 14.37752587481518, 31.132287581699348, 35.7563990692263, 13.674010735607613, 596.9977817012993], 
    'label_type': ['Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Single', 'Multiple'], 
    'connected_components': [78, 438, 1, 1, 1, 136, 314, 1, 1], 
    'local_graph_diameter': [1.0, 4.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 2.0], 
    # 'metric': ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'Accuracy', 'ROC-AUC'], 
    'edge_feature': [0, 0, 0, 0, 0, 0, 0, 0, 8], 
    'dataset': ['cora', 'citeseer', 'pubmed', 'cs', 'physics', 'photo', 'computers', 'arxiv', 'proteins'],
    'local_average_shortest_path_length': [1.0, 2.1947368421052635, 2.251893939393939, 2.4174972314507195, 3.2955307262569833, 2.69797184294598, 2.5689061717099104, 3.0616131441374166, 1.7665342163355409],
    # 'edge_semantic': ['Citation', 'Citation', 'Citation', 'Coauthor', 'Coauthor', 'Coappearance', 'Coappearance', 'Citation', 'Biological Interaction'], 
    'domain': ['Computer Science', 'Computer Science', 'Medical', 'Computer Science', 'Physics', 'Business', 'Business', 'Computer Science', 'Medical'], 
}

class DatasetDescription:
    def __init__(self, descriptions=EXISTING_DESCRIPTION, kgdir='./KG/', load=False):
        self.descriptions = descriptions
        self.datasets = []
        self.uniary_info = None
        self.two_ary_info = None

        self.kgdir = kgdir

        if load:
            with open(self.kg_dir+"KGNAS_knowledge_graph.json", 'r') as f:
                knowledge_graph = json.load(f)

            entities = knowledge_graph['entities']
            relations = knowledge_graph['relations']

            dataset_entities = [entity for entity in entities if entity['type'] == 'dataset']
            dataset_entity_df = {}
            dataset_entity_df['dataset'] = [entity['name'] for entity in dataset_entities]
            property_names = dataset_entities[0]['property'].keys()
            for property in property_names:
                dataset_entity_df[property] = [entity['property'][property] for entity in dataset_entities]
            self.uniary_info = pd.DataFrame(dataset_entity_df)

            dataset_binary_relations = [relation for relation in relations if relation['type'] == 'dataset_related']
            dataset_binary_relation_df = {}
            dataset_binary_relation_df['source_entity'] = [relation['source_entity'] for relation in dataset_binary_relations]
            dataset_binary_relation_df['target_entity'] = [relation['target_entity'] for relation in dataset_binary_relations]
            dataset_binary_relation_df['relation'] = [relation['property']['relation'] for relation in dataset_binary_relations]
            self.two_ary_info = pd.DataFrame(dataset_binary_relation_df)

        if len(self.descriptions) > 0:
            self.prepare_knowledge_graph()
    
    def get_descriptions(self):
        return self.descriptions
    
    def add_description(self, dataset_name, dataset=None,  semantic_description=None, root_dir='datasets/', num_samples=20, num_hops=2, seed=42):
        data_description = GraphDatasetUnderstanding(dataset_name, dataset, semantic_description=semantic_description, num_samples=num_samples, num_hops=num_hops, seed=seed, root_dir=root_dir).process()

        print(data_description)

        for key in self.descriptions.keys():
            if key in data_description.keys():
                self.descriptions[key].append(data_description[key])
            else:
                self.descriptions[key].append(None)

        # self.descriptions[dataset_name] = data_description

        self.datasets.append(dataset_name)

        # print(self.descriptions)

        self.prepare_knowledge_graph()

    
    def prepare_knowledge_graph(self):
        data_description_df = pd.DataFrame(self.descriptions)

        description_names = list(data_description_df.columns)
        # print(description_names)
        description_names.remove('dataset')
        # Uniary information
        self.uniary_info = data_description_df[description_names].select_dtypes(include=['number'])
        self.uniary_info = pd.concat([data_description_df['dataset'], self.uniary_info], axis=1)

        # Two-ary information
        temp_two_ary_info = data_description_df[description_names].select_dtypes(include=['object'])
        self.two_ary_info = pd.DataFrame()
        for col in temp_two_ary_info.columns:
            temp_df = pd.concat([data_description_df['dataset'], temp_two_ary_info[col]], axis=1)
            temp_df['relation'] = 'has_' + col.replace(' ', '_')
            temp_df.rename(columns={col: 'target_entity', 'dataset': 'source_entity'}, inplace=True)
            self.two_ary_info = pd.concat([self.two_ary_info, temp_df])

        # self.two_ary_info = pd.concat([data_description_df[['dataset']], self.two_ary_info], axis=1)
    
    def get_KG_components(self):
        return self.uniary_info, self.two_ary_info
    
    def get_semantic_description_template(self):
        print("The semantic description should be filled into a dictionary, whose keys and suggested values are as follows:")
        print("Please select the most suitbale ONE value from the suggested ones for each key.")

        template_dict = {
            "dataset": "Name of the dataset",
            "num_classes": "Number of classes in the dataset",
            "metric": ["Accuracy", "ROC-AUC"],
            "task": ["Node Classification", "Edge Prediction"],
            "type": ["Homogeneous", "Heterogeneous"],
            "label_type": ["Single", "Multiple"],
            "feature_type": ["Node", "Edge"],
            "edge_semantic": ["Citation", "Coauthor", "Coappearance", "Biological Interaction", "Others"],
            "domain": ["Computer Science", "Medical", "Physics", "Business", "Others"],
        }

        print(template_dict)
        
        return template_dict
    