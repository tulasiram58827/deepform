import glob

import pandas as pd
from pathlib import Path

from spektral.data import Dataset, Graph
from spektral.data.loaders import DisjointLoader
from deepform.data.add_features import read_adjacency
from deepform.common import DATA_DIR, TOKEN_DIR, TRAINING_DIR, TRAINING_INDEX


class GraphDataset(Dataset):
    """
    Graph Dataset for Spektral
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        output = list()
        for file in TRAINING_DIR.glob('*.parquet'):
            try:
                file_name = Path(file).stem
                data = pd.read_parquet(file)
                x = data.loc[:, data.columns != 'label'].to_numpy()
                y = data.loc[:, data.columns == 'label'].to_numpy()
                # if Path(f'{file_name}.graph.npz').exists():
                adj = read_adjacency(f'data/training/416903-collect-files-39738-political-file-2012-non.graph').toarray()
                output.append(Graph(x=x, a=adj, y=y))
            except OSError:
                pass
        return output

dataset = GraphDataset()

# Create a DataLoader
# dataloader = DisjointLoader(dataset, node_level=True, batch_size=8)

# Iterate through DataLoader for training
# for batch in dataloader:
#     print(batch)