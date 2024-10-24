import os
import torch
from torch_geometric.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        super(CustomDataset, self).__init__(root, transform)
        self.file_paths = []
        
        # Recursively collect file paths for .pt files in subdirectories
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith('.pt'):
                    self.file_paths.append(os.path.join(dirpath, filename))
 
    def len(self):
        return len(self.file_paths)
 
    def get(self, idx):
        data = torch.load(self.file_paths[idx])
        if self.transform:
            data = self.transform(data)
        return data.to("cpu")
# Define the root directory containing 'roundabout' and 'intersection' folders
train_directory = '/home/reserachers/adi/Meteor/full_graph_graphs/train'
test_directory = '/home/reserachers/adi/Meteor/full_graph_graphs/test'
# Create an instance of your custom dataset for both 'roundabout' and 'intersection' data
train_dataset = CustomDataset(train_directory)
test_dataset = CustomDataset(test_directory)
# Define batch size
batch_size = 8
 
# Create a single data loader that loads graph data
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
