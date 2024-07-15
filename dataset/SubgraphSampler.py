# Copyright (c) 2024-Current Anonymous
# License: Apache-2.0 license
# This file will handle sampling subgraphs from the dataset.

import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
import random
import os


class SubgraphSampler:
    def __init__(self, dataset, dataset_name, num_samples=20, num_hops=2, seed=42):
        """
        Initialize the SubgraphSampler.
        :param dataset: The graph dataset.
        :param dataset_name: Name of the dataset to be translated.
        :param num_samples: Number of subgraphs to sample.
        :param num_hops: Number of hops to consider in the k-hop subgraph sampling.
        :param seed: Random seed for reproducibility.
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.num_samples = num_samples

        if self.dataset_name in ['ogbn-proteins', 'proteins']:
            self.num_hops = 1
        else:
            self.num_hops = num_hops

        self.seed = seed

    def sample_subgraphs(self, dataset_name, root_dir='datasets/subgraphs/'):
        """
        Samples subgraphs from the dataset, saves each, and returns them as a list.
        :param dataset_name: Name of the dataset for which the subgraphs are sampled.
        :param root_dir: Root directory where the subgraphs will be saved.
        """
        # Ensure the directory exists
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        for i in range(self.num_samples):
            subgraph = self._sample_single_subgraph()
            self._save_subgraph(subgraph, dataset_name, i + 1, root_dir)

    def _sample_single_subgraph(self):
        """
        Samples a single subgraph from the dataset using k-hop sampling.
        :return: A subgraph.
        """
        # Ensure reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            torch.manual_seed(self.seed)

        graph = self.dataset

        # Randomly select a node index
        node_idx = random.randint(0, graph.num_nodes - 1)

        # Get the k-hop subgraph around the selected node
        subgraph_node_idx, subgraph_edge_idx, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=self.num_hops,
            edge_index=graph.edge_index,
            relabel_nodes=True
        )

        # Extract the subgraph
        subgraph_x = graph.x[subgraph_node_idx] if graph.x is not None else None
        subgraph_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None

        # Handle subgraph_y based on whether it's for node-level or graph-level labels
        if graph.y.dim() == 1 and graph.y.numel() == 1:         # Graph-level label
            subgraph_y = graph.y
        else:                                                   # Node-level labels
            subgraph_y = graph.y[subgraph_node_idx] if graph.y is not None else None

        if self.dataset_name in ['ogbn-proteins', 'proteins']:
            subgraph_x = graph.node_species[subgraph_node_idx]
            subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_idx, edge_attr=subgraph_edge_attr, y=subgraph_y)
        else:
            # Create a PyG Data object for the subgraph
            subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_idx, edge_attr=subgraph_edge_attr, y=subgraph_y)

        return subgraph

    def _save_subgraph(self, subgraph, dataset_name, subgraph_number, root_dir):
        """
        Saves a single subgraph to a file.
        :param subgraph: The subgraph to be saved.
        :param dataset_name: Name of the dataset.
        :param subgraph_number: The sequential number of the subgraph.
        :param root_dir: Root directory where the subgraph will be saved.
        """
        file_path = os.path.join(root_dir, f"{dataset_name}_subgraph_{subgraph_number}.pt")
        torch.save(subgraph, file_path)

