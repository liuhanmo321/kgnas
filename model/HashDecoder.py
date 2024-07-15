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

class HashDecoder:
    def __init__(self, gnn_list, gnn_list_proteins):
        self.gnn_list = gnn_list
        self.gnn_list_proteins = gnn_list_proteins

    def decode_hash(self, hash_value, use_proteins=False):
        if use_proteins:
            gnn_g = {i: name for i, name in enumerate(self.gnn_list_proteins)}
            b = len(self.gnn_list_proteins) + 1
        else:
            gnn_g = {i: name for i, name in enumerate(self.gnn_list)}
            b = len(self.gnn_list) + 1
        
        # Determine the lk component from the hash
        lk_hash_values = {
            0: [0, 0, 0, 0],
            1: [0, 0, 0, 1],
            2: [0, 0, 1, 1],
            3: [0, 0, 1, 2],
            4: [0, 0, 1, 3],
            5: [0, 1, 1, 1],
            6: [0, 1, 1, 2],
            7: [0, 1, 2, 2],
            8: [0, 1, 2, 3]
        }
        
        lk_hash = hash_value
        op = []
        lk = None

        # determine the gnn structures
        for i in range(4):
            op.append(gnn_g[lk_hash % b])
            lk_hash = lk_hash // b
        
        # determin the link structure (topology)
        lk = lk_hash_values[lk_hash]
        
        # The list is in reverse order due to the way the hash was constructed
        op.reverse()

        return lk, op


if __name__ == "__main__":
    # Example usage
    hash_value = 10000  # example hash value to decode, which is [0, 0, 0, 1], ['gat', 'gat', 'gat', 'gat']
    use_proteins = False

    decoder = HashDecoder(GNN_LIST, GNN_LIST_PROTEINS)
    lk, op = decoder.decode_hash(hash_value, use_proteins)
    print("lk:", lk)
    print("op:", op)