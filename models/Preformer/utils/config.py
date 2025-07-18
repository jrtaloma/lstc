from argparse import ArgumentParser


def get_arguments():
    parser = ArgumentParser()

    parser.add_argument('--model_name', default='Preformer', type=str, help='Name of the model')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
    parser.add_argument('--experiment_name', default='default', help='Name of the experiment')

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--activation', default='gelu')
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyperparameter for k-means loss')
    parser.add_argument('--beta', type=float, default=0.65, help='The decaying factor for fast simulated annealing')
    parser.add_argument('--temperature', type=int, default=10, help='The initial temperature for fast simulated annealing')
    parser.add_argument('--tol', type=float, default=0.001, help='Tolerance threshold on cluster changes to stop training')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs_pretrain', type=int, default=50, help='Epochs number of the autoencoder pretraining. Set 0 to load pretrained autoencoder')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs number of the model training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate of the model training')

    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--seed', default=0, type=int, help='Seed for the reproducibility of the experiment')

    return parser