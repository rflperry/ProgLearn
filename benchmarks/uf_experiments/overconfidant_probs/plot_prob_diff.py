import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from sklearn.metrics import brier_score_loss


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

n_gaussians = 2
oblique = True
mse = True
if oblique:
    tag = '_oblique'
else:
    tag = ''

data_dir = f'./results_vs_epsilon/'

n = 100
d = 2

# kappa_colors = [
#     '#deebf7',
#     '#c6dbef',
#     '#9ecae1',
#     '#6baed6',
#     '#4292c6',
#     '#2171b5',
#     '#08519c',
# ]

# colors = [
#     '#fdbf6f',
#     '#4daf4a',
#     '#ff7f00',
#     '#a65628',
# ]

colors = {
    'UF (k=0.01)': '#deebf7',
    'UF (k=0.1)': '#c6dbef',
    'UF (k=0.5)': '#9ecae1',
    'UF (k=1)': '#6baed6',
    'UF (k=5)': '#4292c6',
    # '#2171b5',
    # '#08519c',
    'UF (Uncorrected)': '#fdbf6f',
    'SigRF': '#4daf4a',
    'IRF': '#ff7f00',
    'RF': '#a65628',
}


def plot(ax, data, c, mse=False):
    name = data['algo']
    epsilons = data['epsilons']
    y_probs = np.asarray(data['y_probs'])
    y_probs_analytic = np.asarray(data['y_probs_analytic'])
    y_diff = y_probs - y_probs_analytic # overconfidant > 0

    # Select the positive class
    if mse:
        error_pos = np.mean(y_diff[:, :, :, 0]**2, axis=(1, 2))
    else:
        error_pos = np.mean(y_diff[:, :, :, 0], axis=(1, 2))
    ax.plot(epsilons, error_pos, label=name, c=c)
    return ax

f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, c='black', ls=':', label='Calibrated')

for f in os.listdir(data_dir):
    data = unpickle(data_dir + f)
    if f.endswith(f'{n}x{d}' + tag + '.pickle'): # check data and experiment
        ax = plot(ax, data, colors[data['algo']], mse=mse)
        # if f.startswith(f'UF (k'): # Varying kappa have specific colors
        #     ax = plot(ax, data, kappa_colors.pop(), mse=mse)
        # else:
        #     ax = plot(ax, data, colors.pop(), mse=mse)

plt.xlabel('Distance between means')
plt.ylabel('Mean diff ')

# Sort legend
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
if mse:
    plt.savefig(f'./figures/mse_gaussians_{n}x{d}{tag}.pdf')
else:
    plt.savefig(f'./figures/mean_error_gaussians_{n}x{d}{tag}.pdf')
