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

oblique = False
score = 'accuracy' #'mse', 'diff', 'accuracy', 'brier'
if oblique:
    tag = '_oblique'
else:
    tag = ''

data_dir = f'./results_vs_epsilon/'

n = 100
d = 2


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


def plot(ax, name, data, score, c):
    epsilons = data['epsilons']
    y = np.asarray(data['y'])
    y_probs = np.asarray(data['y_probs'])
    y_probs_analytic = np.asarray(data['y_probs_analytic'])
    y_diff = y_probs - y_probs_analytic # overconfidant > 0

    # Select the positive class
    # Select one of the classes, 1st here WLOG
    if score == 'mse':
        error = np.mean(y_diff[:, :, :, 1]**2, axis=(1, 2))
        # stds = [np.std(np.mean(np.stack(y_diff[i])[:, :, 1]**2, axis=1)) for i in range(len(ns))]
    elif score == 'diff':
        error = np.mean(y_diff[:, :, :, 1], axis=(1, 2))
    elif score == 'accuracy':
        error = np.mean(np.abs(np.argmax(y_probs, axis=-1) - y), axis=(1, 2))
    elif score == 'brier':
        score = y_probs
        score[np.arange(len(y_probs)), y.astype(int)] -= 1
        error = np.mean(np.sum(scores**2, axis=1))
    # if mse:
    #     error_pos = np.mean(y_diff[:, :, :, 1]**2, axis=(1, 2))
    # else:
    #     error_pos = np.mean(y_diff[:, :, :, 1], axis=(1, 2))
    ax.plot(epsilons, error, label=name, c=c)
    return ax

f, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.axhline(0, c='black', ls=':', label='Calibrated')

for f in os.listdir(data_dir):
    if f.endswith(f'{n}x{d}' + tag + '.pickle'): # check data and experiment
        data = unpickle(data_dir + f)
        name = data['algo']
        if name.startswith('UF') and not name.endswith(')'):
            name = name + ')'
        ax = plot(ax, name, data, score, colors[name])
        # if f.startswith(f'UF (k'): # Varying kappa have specific colors
        #     ax = plot(ax, data, kappa_colors.pop(), mse=mse)
        # else:
        #     ax = plot(ax, data, colors.pop(), mse=mse)

plt.xlabel('Distance between means')
if score == 'mse':
    plt.ylabel('Mean squared probability error')
elif score == 'diff':
    plt.ylabel('Mean probability difference')
elif score == 'accuracy':
    plt.ylabel('Mean 0-1 Error')
elif score == 'brier':
    plt.ylabel('Brier Score')

# Sort legend
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'./figures/{score}_gaussians_{n}x{d}{tag}.pdf')
