from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset

class DatasetReader:
    def __init__(self, dataset_name, root_dir, local_dataset=False, local_path=None):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.local_dataset = local_dataset
        self.local_path = local_path
        

    def read_dataset(self):
        if self.dataset_name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=self.root_dir, name=self.dataset_name)
        elif self.dataset_name in ['cs', 'physics']:
            dataset = Coauthor(root=self.root_dir, name=self.dataset_name)
        elif self.dataset_name in ['photo', 'computers']:
            dataset = Amazon(root=self.root_dir, name=self.dataset_name)
        elif self.dataset_name in ['arxiv', 'proteins']:
            dataset = PygNodePropPredDataset(name=f'ogbn-{self.dataset_name}', root=self.root_dir)
        else:
            dataset = None
        return dataset[0]