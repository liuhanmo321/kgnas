# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file will serve as the main module to orchestrate the understanding of graph datasets.

from .DatasetReader import DatasetReader
from .SubgraphSampler import SubgraphSampler
from .GraphTranslator import GraphTranslator
# from .DescriptionCombiner import DescriptionCombiner
import json
import os

METRICS = ['local_average_shortest_path_length', 'local_graph_diameter',
            'local_average_closeness_centrality', 'local_average_betweenness_centrality',
            'node_count', 'edge_count', 'average_degree',
            'density', 'average_clustering_coefficient',
            'connected_components', 'assortativity',
            'average_degree_centrality', 'average_eigenvector_centrality']

class GraphDatasetUnderstanding:
    def __init__(self, dataset_name, dataset=None, semantic_description=None, metrics_list=METRICS, root_dir='datasets/',
                 local_dataset=False, local_path=None, predefined_descriptions_path='datasets/Benchmark datasets/predefined_descriptions.json',
                 num_samples=20, num_hops=2, seed=42):
        """
        Initialize the GraphDatasetUnderstanding module.
        :param dataset_name: Name of the dataset to be loaded.
        :param user_description: Textual description provided by the user.
        :param metrics_list: List of formats for translating the graph data.
        :param root_dir: Root directory for datasets.
        :param local_dataset: Flag indicating if the dataset is a local one.
        :param local_path: Path to the local dataset.
        :param predefined_descriptions_path: Path to the JSON file with predefined descriptions.
        :param num_samples: Number of subgraphs to sample.
        :param num_hops: Number of hops to consider in the k-hop subgraph sampling.
        :param seed: Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.local_dataset = local_dataset
        self.local_path = local_path
        self.metrics_list = metrics_list or []
        # self.no_statistics = no_statistics
        # self.use_semantic = use_semantic
        if dataset is None:
            self.dataset_reader = DatasetReader(dataset_name, root_dir, local_dataset, local_path)
            self.dataset = self.dataset_reader.read_dataset()
            print("dataset successfully read")
        else:
            self.dataset = dataset
        # if self.dataset_reader.specific_dataset is not None:
        #     self.dataset_name = self.dataset_reader.specific_dataset
        # else:
        self.dataset_name = dataset_name
        self.semantic_description = semantic_description
        self.statistical_description = None
        self.combined_description = None

        # Load predefined descriptions
        # with open(predefined_descriptions_path, 'r') as file:
        #     self.predefined_descriptions = json.load(file)

        # default_msg = "The user does not provide a description for the dataset, please understand it based entirely " \
        #               "on the following graph description language techniques."
        # self.user_description = user_description or self.predefined_descriptions.get(self.dataset_name, default_msg)
        self.num_samples = num_samples
        self.num_hops = num_hops
        self.seed = seed

    def process(self):
        """
        Processes the graph dataset to understand it through various descriptions.
        :return: Combined description of the dataset suitable for LLM prompting.
        """
        # if self.no_statistics or len(self.metrics_list) == 0:
        #     combined_description_path = f"{self.root_dir}{self.dataset_name}/user_description.txt"
        # else:
            
        combined_description_path = f"{self.root_dir}{self.dataset_name}/combined_description.json"
        
        # Check if combined description is already stored locally for public datasets
        # if os.path.exists(combined_description_path):
        #     with open(combined_description_path, 'r') as file:
        #         combined_description = file.read()
        #     # combined_description = self.filter_description(combined_description)
        # else:
            # If not stored locally, follow the processing steps
        combined_description = self._process_dataset()

        # Save the combined description for public datasets
        #if self.dataset_name in self.predefined_descriptions:
        #    with open(combined_description_path, 'w') as file:
        #        file.write(combined_description)
        with open(combined_description_path, 'w') as file:
            json.dump(combined_description, file)
            # file.write(combined_description)

        return combined_description

    def _process_dataset(self):
        """
        Helper method to process the dataset through submodules.
        :return: Combined description of the dataset.
        """
        # if not self.no_statistics:
            # Step 1: Read Dataset
        # dataset = self.dataset_reader.read_dataset()

        # Step 2: Sample Subgraphs
        subgraphs_folder = f"{self.root_dir}{self.dataset_name}/subgraphs/"
        sampler = SubgraphSampler(self.dataset, self.dataset_name, self.num_samples, self.num_hops, self.seed)
        sampler.sample_subgraphs(self.dataset_name, subgraphs_folder)

        # Step 3: Translate Subgraphs
        translator = GraphTranslator(self.dataset, self.dataset_name, subgraphs_folder, self.metrics_list)
        self.statistical_description = translator.translate()
        self.print_formatted_translations(self.statistical_description)

        # Step 4: Combine Descriptions
        # is_bench = 'Benchmark' in self.root_dir
        # combiner = DescriptionCombiner(self.dataset_name, self.user_description, is_bench, self.num_hops,
        #                                 translations)
        # combined_description = combiner.combine_descriptions()
        self.combined_description = self.semantic_description | self.statistical_description
        
        # else:
        #     # Step 1: Combine Descriptions
        #     is_bench = 'Benchmark' in self.root_dir
        #     combiner = DescriptionCombiner(self.dataset_name, self.user_description, is_bench, self.num_hops, None)
        #     combined_description = combiner.combine_descriptions()

        return self.combined_description

    def print_formatted_translations(self, translations):
        """
        Prints the graph metrics formatted into categories for better readability.

        Args:
        - translations (dict): A dictionary containing various graph metrics,
          with keys indicating the metric name and values indicating the metric value.
        """

        # Splitting the dictionary for clearer categorization
        # Extract local metrics for k-hop subgraphs
        local_metrics = {key: translations[key] for key in translations if 'local_' in key}

        # Extract general metrics for the overall graph
        general_metrics = {key: translations[key] for key in translations if 'local_' not in key}

        # Printing local metrics
        print(f"{self.num_hops}-Hop Subgraph Metrics:")
        print("-----------------------")
        for key, value in local_metrics.items():
            # Removing 'local_' from the key for printing
            formatted_key = key.replace('local_', '').replace('_', ' ').title()
            print(f"- {formatted_key}: {value}")

        # Adding a space between the sections for better readability
        print("\nGeneral Graph Metrics:")
        print("----------------------")
        for key, value in general_metrics.items():
            # Formatting the key to make it more readable
            formatted_key = key.replace('_', ' ').title()
            print(f"- {formatted_key}: {value}")
    
    def filter_description(self, description):

        metrics_list = [metric.lower().replace('local_', '') for metric in self.metrics_list]
        local_metrics = {
            'local_average_shortest_path_length',
            'local_graph_diameter',
            'local_average_closeness_centrality',
            'local_average_betweenness_centrality'
        }

        # Splitting the content into lines
        lines = description.split('\n')

        # Variable to keep track of whether we are in a metrics section
        in_metrics_section = False
        in_user_description = False

        # List to hold the filtered content
        filtered_content = []

        for line in lines:
            if not self.use_semantic and 'User Description:' in line:
                in_metrics_section = False
                in_user_description = True

            '''
            if 'User Description:' in line:
                # Start skipping lines from this point
                in_user_description = True
            elif in_user_description and line.strip() == '':
                # If we reach an empty line, we assume it's the end of the User Description section
                in_user_description = False
            elif not in_user_description:
            '''
            # Check if we are entering a metrics section
            if 'Subgraph Metrics:' in line:
                in_metrics_section = True
                in_user_description = False
                if any(metric in self.metrics_list for metric in local_metrics):
                    filtered_content.append(line)
            elif 'Graph Metrics:' in line:
                in_metrics_section = True
                in_user_description = False
                if any(metric not in local_metrics for metric in self.metrics_list):
                    filtered_content.append(line)
            # Check if we are exiting a metrics section
            elif in_metrics_section and line.strip() == '':
                in_metrics_section = False
                filtered_content.append(line)
            # Process metrics lines if we are in a metrics section
            elif in_metrics_section:
                # Extract the metric name from the line
                metric_name = line.lstrip('-').strip().split(':')[0].strip().lower().replace(' ', '_').replace('-', '')

                # If the metric name is in the allowed list, add it to the filtered content
                if metric_name in metrics_list:
                    filtered_content.append(line)
            else:
                if not self.use_semantic and in_user_description:
                    continue
                # If we are not in a metrics section, just add the line
                filtered_content.append(line)

        return '\n'.join(filtered_content)

