import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='DNFCS', type=str, help='Name of the model')
    parser.add_argument('--dataset_name', required=True, help='Name of the dataset')
    parser.add_argument('--experiment_name', default='default', help='Name of the experiment')

    parser.add_argument('--n_z', type=int, default=128, help='Size of the hidden state')
    parser.add_argument('--beta', type=float, default=0.001, help="Hyperparameter for the balance of the within-cluster distance and the between-cluster distance in the cluster space")
    parser.add_argument('--m', type=float, default=2.0, help='Fuzzifier')
    parser.add_argument('--alpha_fcs', type=float, default=0.01, help='Hyperparameter for FCS loss')
    parser.add_argument('--alpha_kl', type=float, default=0.1, help='Hyperparameter for KL divergence loss')
    parser.add_argument('--tol', type=float, default=0.001, help='Tolerance threshold on cluster changes to stop training')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs_pretrain', type=int, default=50, help='Epochs number of the autoencoder pretraining. Set 0 to load pretrained autoencoder')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum epochs number of the model training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of the model training')
    parser.add_argument('--step_size', default=5, type=int, help='Decay the learning rate of each parameter group every step_size epochs')

    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--seed', default=0, type=int, help='Seed for the reproducibility of the experiment')

    return parser