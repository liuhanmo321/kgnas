# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file is responsible for translating subgraphs to a graph description language.

import os
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx


class GraphTranslator:
    def __init__(self, dataset, dataset_name, subgraphs_folder, metrics_list):
        """
        Initializes the GraphTranslator with the dataset, path to the folder containing saved subgraphs,
        and a list of metrics to calculate.

        Parameters:
        - dataset: The PyTorch Geometric dataset from which subgraphs were sampled.
        - dataset_name: Name of the dataset to be translated.
        - subgraphs_folder: Path to the folder containing saved subgraphs.
        - metrics_list: List of metrics to calculate.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.subgraphs_folder = subgraphs_folder
        self.metrics_list = metrics_list

        # Predefined categorization of metrics into intensive and non-intensive
        self.intensive_metrics_names = ['local_average_shortest_path_length', 'local_graph_diameter',
                                        'local_average_closeness_centrality', 'local_average_betweenness_centrality']
        self.non_intensive_metrics_names = ['node_count', 'edge_count', 'average_degree',
                                            'density', 'average_clustering_coefficient',
                                            'connected_components', 'assortativity',
                                            'average_degree_centrality', 'average_eigenvector_centrality']

        if self.dataset_name in ['ogbn-proteins', 'proteins']:
            # Move 'average_clustering_coefficient' from non-intensive to intensive list
            self.non_intensive_metrics_names.remove('average_clustering_coefficient')
            self.intensive_metrics_names.append('average_clustering_coefficient')

        # Filter the requested metrics into intensive and non-intensive based on the provided list
        self.intensive_metrics = [m for m in self.intensive_metrics_names if m in metrics_list]
        self.non_intensive_metrics = [m for m in self.non_intensive_metrics_names if m in metrics_list]

    def translate(self):
        """
        Computes the specified metrics, distinguishing between intensive and non-intensive metrics as defined.
        """
        # Check if the dataset has edge attributes
        has_edge_attr = 'edge_attr' in self.dataset.keys()

        # Convert the entire dataset to a NetworkX graph
        if has_edge_attr:
            if self.dataset_name in ['ogbn-proteins', 'proteins']:
                full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['node_species'],
                                         edge_attrs=['edge_attr'])
            else:
                full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['x'],
                                         edge_attrs=['edge_attr'])
        else:
            # If no edge attributes, do not specify edge_attrs in the conversion
            full_graph = to_networkx(self.dataset, to_undirected=True, node_attrs=['x'])

        # Calculate and print non-intensive metrics
        results = {}
        results['node_feature'] = self.dataset.x.size(1) if self.dataset.x is not None else 0
        results['edge_feature'] = self.dataset.edge_attr.size(1) if self.dataset.edge_attr is not None else 0

        for metric in self.non_intensive_metrics:
            results[metric] = getattr(self, f"compute_{metric}")(full_graph)
            # print(metric)

        # Load subgraphs and compute intensive metrics
        intensive_results = {metric: [] for metric in self.intensive_metrics}
        for path in self.list_subgraph_files():
            subgraph = self.load_subgraph(path)
            for metric in self.intensive_metrics:
                metric_value = getattr(self, f"compute_{metric}")(subgraph)
                intensive_results[metric].append(metric_value)

        # Aggregate results for intensive metrics
        for metric, values in intensive_results.items():
            results[metric] = np.nanmean(values)        # Using nanmean to safely handle any NaN values

        if self.dataset_name in ['ogbn-proteins', 'proteins'] and 'average_clustering_coefficient' in results:
            # Change the key in the results dictionary
            results['local_average_clustering_coefficient'] = results.pop('average_clustering_coefficient')

        return results

    def list_subgraph_files(self):
        """
        Lists all .pt files in the subgraphs_folder.
        :return: A list of full paths to the subgraph files.
        """
        return [os.path.join(self.subgraphs_folder, f) for f in os.listdir(self.subgraphs_folder) if f.endswith('.pt')]

    def load_subgraph(self, file_path):
        """
        Loads a saved subgraph from a file and converts it to a NetworkX graph object.
        It also handles both node-level and graph-level labels.
        :param file_path: The path to the file where the subgraph is saved.
        :return: A NetworkX graph object representing the subgraph, with node and edge features, and possibly
        graph-level labels.
        """
        # Load the PyTorch Geometric Data object from file
        subgraph = torch.load(file_path)

        # Check if the subgraph has edge attributes
        has_edge_attr = 'edge_attr' in subgraph.keys()

        # Convert the entire dataset to a NetworkX graph
        if has_edge_attr:
            G = to_networkx(subgraph, to_undirected=True, node_attrs=['x'], edge_attrs=['edge_attr'])
        else:
            # If no edge attributes, do not specify edge_attrs in the conversion
            G = to_networkx(subgraph, to_undirected=True, node_attrs=['x'])

        # If 'y' exists and is graph-level (1D tensor), add it as a graph attribute
        if 'y' in subgraph and subgraph.y.dim() == 1 and subgraph.y.numel() == 1:
            G.graph['label'] = subgraph.y.item()        # Assuming it's a single label

        # If 'y' exists and is node-level, add it as node attributes
        #elif 'y' in subgraph and subgraph.y.dim() > 1:
        elif 'y' in subgraph:
            for i, node_data in enumerate(subgraph.y):
                # Assuming nodes are relabeled from 0 to N-1 in the subgraph
                if self.dataset_name in ['ogbn-proteins', 'proteins']:
                    G.nodes[i]['label'] = node_data.tolist()
                else:
                    G.nodes[i]['label'] = node_data.item()

        return G

    @staticmethod
    def compute_node_count(nx_graph):
        """Computes the node count of the graph."""
        return nx_graph.number_of_nodes()

    @staticmethod
    def compute_edge_count(nx_graph):
        """Computes the edge count of the graph."""
        return nx_graph.number_of_edges()

    @staticmethod
    def compute_density(nx_graph):
        """Computes the density of the graph."""
        return nx.density(nx_graph)

    @staticmethod
    def compute_average_degree(nx_graph):
        """Computes the average degree of the graph."""
        total_degree = sum(dict(nx_graph.degree()).values())
        avg_degree = total_degree / nx_graph.number_of_nodes()
        return avg_degree

    @staticmethod
    def compute_average_clustering_coefficient(nx_graph):
        """Computes the average clustering coefficient of the graph."""
        return nx.average_clustering(nx_graph)

    @staticmethod
    def compute_connected_components(nx_graph):
        """Computes the number of connected components in the graph."""
        return nx.number_connected_components(nx_graph)

    @staticmethod
    def compute_assortativity(nx_graph):
        """Computes the degree assortativity coefficient of the graph."""
        return nx.degree_assortativity_coefficient(nx_graph)

    @staticmethod
    def compute_average_degree_centrality(nx_graph):
        """Computes the average degree centrality of the graph."""
        degree_centrality = nx.degree_centrality(nx_graph)
        return np.mean(list(degree_centrality.values()))

    @staticmethod
    def compute_average_eigenvector_centrality(nx_graph):
        """Computes the average eigenvector centrality of the graph."""
        eigenvector_centrality = nx.eigenvector_centrality(nx_graph, max_iter=1000)
        return np.mean(list(eigenvector_centrality.values()))

    @staticmethod
    def compute_local_average_shortest_path_length(nx_graph):
        """Computes the average shortest path length for the largest connected component of the graph."""
        if nx.is_connected(nx_graph):
            return nx.average_shortest_path_length(nx_graph)
        else:
            largest_cc = max(nx.connected_components(nx_graph), key=len)
            subgraph = nx_graph.subgraph(largest_cc)
            return nx.average_shortest_path_length(subgraph)

    @staticmethod
    def compute_local_graph_diameter(nx_graph):
        """Computes the diameter for the largest connected component of the graph."""
        if nx.is_connected(nx_graph):
            return nx.diameter(nx_graph)
        else:
            largest_cc = max(nx.connected_components(nx_graph), key=len)
            subgraph = nx_graph.subgraph(largest_cc)
            return nx.diameter(subgraph)

    @staticmethod
    def compute_local_average_closeness_centrality(nx_graph):
        """Computes the average closeness centrality of the graph."""
        closeness_centrality = nx.closeness_centrality(nx_graph)
        return np.mean(list(closeness_centrality.values()))

    @staticmethod
    def compute_local_average_betweenness_centrality(nx_graph):
        """Computes the average betweenness centrality of the graph."""
        betweenness_centrality = nx.betweenness_centrality(nx_graph)
        return np.mean(list(betweenness_centrality.values()))

    '''
    def translate(self):
        """
        Main function to compute graph metrics. Computes non-intensive metrics directly on the dataset and
        averages intensive metrics over sampled subgraphs.
        """
        # Convert the entire dataset to a NetworkX graph for non-intensive metrics
        full_graph = to_networkx(self.dataset[0], to_undirected=True)
        non_intensive_metrics = self.compute_non_intensive_metrics(full_graph)
        for metric, value in non_intensive_metrics.items():
            print(f"{metric}: {value}")

        # Load subgraphs, compute intensive metrics, and average them
        for subgraph_path in self.sampled_subgraphs_paths:
            nx_subgraph = self.load_subgraph(subgraph_path)
            intensive_metrics = self.compute_intensive_metrics(nx_subgraph)
            # TBD

        return aggregated_metrics
    
    def compute_non_intensive_metrics(self, nx_graph):
        """
        Computes various graph metrics for a given NetworkX graph.

        Parameters:
        - nx_graph: A NetworkX graph object.

        Returns:
        - A dictionary containing computed metrics for the subgraph.
        """
        metrics = {
            'node_count': nx_graph.number_of_nodes(),
            'edge_count': nx_graph.number_of_edges(),
            'average_degree': sum(dict(nx_graph.degree()).values()) / nx_graph.number_of_nodes(),
            'density': nx.density(nx_graph),
            'average_clustering_coefficient': nx.average_clustering(nx_graph),
            'connected_components': nx.number_connected_components(nx_graph),
            'assortativity': nx.degree_assortativity_coefficient(nx_graph),
            'average_degree_centrality': np.mean(list(nx.degree_centrality(nx_graph).values())),
            'average_eigenvector_centrality': np.mean(list(nx.eigenvector_centrality(nx_graph, max_iter=1000).values()))
        }
        return metrics

    def compute_intensive_metrics(self, nx_graph):
        """
        Computes various graph metrics for a given NetworkX graph (subgraph).

        Parameters:
        - nx_graph: A NetworkX graph object.

        Returns:
        - A dictionary containing computed metrics for the subgraph.
        """
        metrics = {
            'average_shortest_path_length': nx.average_shortest_path_length(nx_graph),
            'graph_diameter': nx.diameter(nx_graph),
            'average_closeness_centrality': np.mean(list(nx.closeness_centrality(nx_graph).values())),
            'average_betweenness_centrality': np.mean(list(nx.betweenness_centrality(nx_graph).values()))
        }
        return metrics

    def load_subgraph(self, file_path):
        """
        Loads a subgraph from a JSON file and reconstructs it into both PyTorch Geometric and NetworkX formats.

        Parameters:
        - file_path (str): The path to the JSON file containing the subgraph data.

        Returns:
        - A tuple containing:
            - PyTorch Geometric Data object of the subgraph.
            - NetworkX graph object of the same subgraph.
        """
        # Load the subgraph data from the JSON file
        with open(file_path, 'r') as file:
            json_data = json.load(file)

        # Reconstruct the NetworkX graph
        nx_graph = nx.Graph()
        for edge in json_data['edge_index']:
            nx_graph.add_edge(edge[0], edge[1])

        if 'node_features' in json_data:
            for idx, features in enumerate(json_data['node_features']):
                nx_graph.nodes[idx]['features'] = np.array(features)

        if 'node_labels' in json_data:
            for idx, label in enumerate(json_data['node_labels']):
                nx_graph.nodes[idx]['label'] = label

        return nx_graph
    '''

    '''
    # Compute non-intensive metrics directly on the entire dataset (placeholder)
        print("Computing non-intensive metrics on the entire dataset...")

        # For intensive metrics, compute on each subgraph and take the average
        intensive_metrics_list = []
        for subgraph_path in self.sampled_subgraphs_paths:
            nx_subgraph = self.load_subgraph(subgraph_path)
            subgraph_metrics = self.compute_metrics_for_subgraph(nx_subgraph)
            intensive_metrics_list.append(subgraph_metrics)

        # Aggregate metrics across all subgraphs (taking the mean, handling NaNs as needed)
        aggregated_metrics = {
            'average_shortest_path_length': np.nanmean(
                [m['average_shortest_path_length'] for m in intensive_metrics_list]),
            'graph_diameter': np.nanmax([m['graph_diameter'] for m in intensive_metrics_list]),
            # Max, since diameter is a max metric
            'average_eigenvector_centrality': np.nanmean(
                [m['average_eigenvector_centrality'] for m in intensive_metrics_list])
        }

        print("Aggregated Metrics:", aggregated_metrics)
        return aggregated_metrics
    
    def translate(self, formats):
        """
        Translates the subgraph to the specified formats.
        :param formats: A list of formats to translate the subgraph into (e.g., ['gml', 'edge_list']).
        :return: A dictionary of subgraph representations in the specified formats.
        """
        translations = {}
        #g = self._convert_to_networkx()
        g = pyg_utils.to_networkx(self.subgraph, to_undirected=True)

        if 'gml' in formats:
            translations['gml'] = self._translate_to_gml(g)

        if 'graphml' in formats:
            translations['graphml'] = self._translate_to_graphml(g)

        if 'edge_list' in formats:
            translations['edge_list'] = self._translate_to_edge_list(g)

        if 'adjacency_list' in formats:
            translations['adjacency_list'] = self._translate_to_adjacency_list(g)

        return translations
    
    def _translate_to_gml(self, g):
        """
        Translates a NetworkX graph to Graph Modeling Language (GML) format.
        :param g: A NetworkX graph.
        :return: A string representation of the graph in GML format.
        """
        output = io.StringIO()
        nx.write_gml(g, output)
        return output.getvalue()

    def _translate_to_graphml(self, g):
        """
        Translates a NetworkX graph to Graph Markup Language (GraphML) format.
        :param g: A NetworkX graph.
        :return: A string representation of the graph in GraphML format.
        """
        output = io.StringIO()
        nx.write_graphml(g, output)
        return output.getvalue()

    def _translate_to_edge_list(self, g):
        """
        Translates a NetworkX graph to an edge list format.
        :param g: A NetworkX graph.
        :return: A string representation of the graph as an edge list.
        """
        output = io.StringIO()
        nx.write_edgelist(g, output, data=False)
        return output.getvalue()

    def _translate_to_adjacency_list(self, g):
        """
        Translates a NetworkX graph to an adjacency list format.
        :param g: A NetworkX graph.
        :return: A string representation of the graph as an adjacency list.
        """
        return '\n'.join(['%s: %s' % (node, ' '.join(map(str, neighbors)))
                          for node, neighbors in nx.generate_adjlist(g)])

    def _convert_to_networkx(self):
        """
        Converts the PyG subgraph to a NetworkX graph.
        :return: A NetworkX graph object representing the subgraph.
        """
        g = nx.Graph()
        edge_index = self.subgraph['edge_index'].t().tolist()
        g.add_edges_from(edge_index)

        # Check if node features are available and add them
        if 'x' in self.subgraph and self.subgraph['x'] is not None:
            for i, features in enumerate(self.subgraph['x']):
                # Assuming features are a tensor, convert them to a list
                g.nodes[i]['features'] = features.tolist()

        # Check if node labels are available and add them
        if 'y' in self.subgraph and self.subgraph['y'] is not None:
            for i, label in enumerate(self.subgraph['y']):
                # Assuming labels are a tensor or a list, assign them directly
                g.nodes[i]['label'] = label

        return g

    def translate_with_graph_enhanced_llm(self):
        """
        Placeholder for future implementation with graph-enhanced LLMs or neural network models.
        This function would use a pre-trained model to generate a textual description of the graph.
        :return: A placeholder string indicating where the LLM output would go.
        """
        # Example: Use a pre-trained model to generate a textual description of the graph
        return "Graph description generated by graph-enhanced LLM (Placeholder)"
    '''

