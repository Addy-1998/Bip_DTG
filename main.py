
import argparse
from train_test_loop import train_and_test
from config import get_config

def main():
    parser = argparse.ArgumentParser(description='Run the training and testing loop with configurable parameters.')
    parser.add_argument('--train_directory', type=str, help='Path to training dataset directory.', default=get_config()['train_directory'])
    parser.add_argument('--test_directory', type=str, help='Path to testing dataset directory.', default=get_config()['test_directory'])
    parser.add_argument('--batch_size', type=int, help='Batch size for training and testing.', default=get_config()['batch_size'])
    parser.add_argument('--learning_rate', type=float, help='Initial learning rate for training.', default=get_config()['learning_rate'])
    parser.add_argument('--num_epochs', type=int, help='Number of epochs for training.', default=get_config()['num_epochs'])
    parser.add_argument('--num_heads', type=int, help='Number of heads in GAT layers.', default=get_config()['num_heads'])
    parser.add_argument('--values_of_k', type=int, help='Value of PageRank mechanism.', default=get_config()['values_of_k'])
    
    # Parse arguments
    args = parser.parse_args()
    config = get_config()
    # Update configuration with arguments if provided
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    # Run training and testing
    train_and_test(config)

if __name__ == '__main__':
    main()
