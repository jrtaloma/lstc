import os
import pandas as pd
import numpy as np


path_results = './results.csv'


model_name = 'IDEC'


experiment_name = 'default'


dataset_names = [
    'CinCECGTorso',
    'NonInvasiveFetalECGThorax1',
    'NonInvasiveFetalECGThorax2',
    'StarLightCurves',
    'UWaveGestureLibraryX',
    'UWaveGestureLibraryY',
    'MixedShapesRegularTrain',
    'MixedShapesSmallTrain',
    'EOGVerticalSignal',
    'SemgHandMovementCh2',
    'ECG5000',
    'OSULeaf',
    'Symbols',
    'MiddlePhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxTW',
    'SyntheticControl'
]


seeds = [0, 1, 2, 3, 4]


if __name__ == '__main__':

    metrics = {
        'Dataset': [],
        'RI_mean': [],
        'RI_std': [],
        'ARI_mean': [],
        'ARI_std': [],
        'NMI_mean': [],
        'NMI_std': [],
        'Epoch_mean': [],
        'Epoch_std': []
    }

    for dataset_name in dataset_names:
        ri_scores, ari_scores, nmi_scores, epochs = [], [], [], []
        for seed in seeds:
            path_logs = os.path.join('./logs', 'UCRArchive_2018', dataset_name, experiment_name)
            path_csv = os.path.join(path_logs, f'metrics_{model_name}_{seed}.csv')
            df = pd.read_csv(path_csv)
            ri = df['RI'].values[0]
            ari = df['ARI'].values[0]
            nmi = df['NMI'].values[0]
            epoch = df['Epoch'].values[0]
            ri_scores.append(ri)
            ari_scores.append(ari)
            nmi_scores.append(nmi)
            epochs.append(epoch)
        avg_ri, std_ri = np.mean(ri_scores), np.std(ri_scores)
        avg_ari, std_ari = np.mean(ari_scores), np.std(ari_scores)
        avg_nmi, std_nmi = np.mean(nmi_scores), np.std(nmi_scores)
        avg_epoch, std_epoch = np.mean(epochs), np.std(epochs)

        metrics['Dataset'].append(dataset_name)
        metrics['RI_mean'].append(avg_ri)
        metrics['RI_std'].append(std_ri)
        metrics['ARI_mean'].append(avg_ari)
        metrics['ARI_std'].append(std_ari)
        metrics['NMI_mean'].append(avg_nmi)
        metrics['NMI_std'].append(std_nmi)
        metrics['Epoch_mean'].append(avg_epoch)
        metrics['Epoch_std'].append(std_epoch)

    df_results = pd.DataFrame.from_dict(metrics)
    print(df_results)

    print(f'RI (avg): {np.mean(metrics["RI_mean"])}')
    print(f'ARI (avg): {np.mean(metrics["ARI_mean"])}')
    print(f'NMI (avg): {np.mean(metrics["NMI_mean"])}')
    print(f'Epoch (avg): {np.mean(metrics["Epoch_mean"])}')
