
# Configuration parameters for the model and training process
config = {
    'train_directory': '/path/to/train/dataset',
    'test_directory': '/path/to/test/dataset',
    'batch_size': 8,
    'learning_rate': 0.01,
    'num_epochs': 100,
    'num_heads': 4,  # GAT heads
    'values_of_k': 10,  # Values of K in K-nearest neighbors or similar
}

def get_config():
    return config
