import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='DTCR', type=str, help='Name of the model')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
    parser.add_argument('--experiment_name', default='default', help='Name of the experiment')

    parser.add_argument('--hidden_sizes', type=list, default=[100,50,50], help='Number of hidden units of layers')
    parser.add_argument('--dilations', type=list, default=[1,4,16], help='Dilation of layers')
    parser.add_argument('--cell_type', default='LSTM', help='Cell type for RNN')
    parser.add_argument('--weight', type=float, default=1.0, help='Hyperparameter for K-means loss')
    parser.add_argument('--tol', type=float, default=0.001, help='Tolerance threshold on cluster changes to stop training')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs number of the model training')
    parser.add_argument('--lr', type=float, default=5e-3, help='Learning rate of the model training')
    parser.add_argument('--alter_iter', type=int, default=10, help='Number of epochs between two updates of the cluster indicator matrix')

    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--seed', default=0, type=int, help='Seed for the reproducibility of the experiment')

    return parser