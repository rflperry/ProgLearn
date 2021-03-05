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

score = 'mse' #'mse', 'diff', 'accuracy', 'brier'
tag = '_trunk'

data_dir = f'./results_vs_n/'

d = 200
epsilon = 1

colors = {
    'UF (k=0.01)': '#deebf7',
    'UF (k=0.1)': '#c6dbef',
    # 'UF (k=0.5)': '#9ecae1',
    'UF (k=1)': '#6baed6',
    # 'UF (k=5)': '#4292c6',
    'UF (k=10)': '#2171b5',
    # '#08519c',
    'UF (Uncorrected)': '#fdbf6f',
    'SigRF': '#4daf4a',
    'IRF': '#ff7f00',
    'RF': '#a65628',
}


def plot(ax, data, c, score):
    name = data['algo']
    ns = 2*np.asarray(data['sample_sizes']) # since samples n from each of 2 classes
    y_probs = np.asarray(data['y_probs'])
    y_probs_analytic = np.asarray(data['y_probs_analytic'])
    y_diff = y_probs - y_probs_analytic # overconfidant > 0
    y = data['y']

    # Select one of the classes, 1st here WLOG
    if score == 'mse':
        error = [np.mean(np.stack(y_diff[i])[:, :, 1]**2) for i in range(len(ns))]
        stds = [np.std(np.mean(np.stack(y_diff[i])[:, :, 1]**2, axis=1)) for i in range(len(ns))]
    elif score == 'diff':
        error = [np.mean(np.stack(y_diff[i])[:, :, 1]) for i in range(len(ns))]
    elif score == 'accuracy':
        error = [np.mean(np.mean((np.argmax(np.stack(y_probs[i]), axis=2) - y[i])**2, axis=1)) for i in range(len(ns))]
        stds = [np.std(np.mean((np.argmax(np.stack(y_probs[i]), axis=2) - y[i])**2, axis=1)) for i in range(len(ns))]
    elif score == 'brier':
        error = [np.mean(np.mean((np.stack(y_probs[i])[:, :, 1] - y[i])**2, axis=1)) for i in range(len(ns))] # positive class
        stds = [np.std(np.mean((np.stack(y_probs[i])[:, :, 1] - y[i])**2, axis=1)) for i in range(len(ns))]
    # ax.plot(ns, error, label=name, c=c)
    plt.errorbar(ns, error, label=name, c=c, yerr=stds, capsize=5)
    # stds = np.asarray(stds)
    # error = np.asarray(error)
    # plt.fill_between(ns, error-stds, error+stds, color=c, alpha=0.1, label=name)
    return ax

f, ax = plt.subplots(1, 1, figsize=(8, 5))
# ax.axhline(0, c='black', ls=':', label='Calibrated')

for f in os.listdir(data_dir):
    data = unpickle(data_dir + f)
    if f.endswith(f'{epsilon}-{d}' + tag + '.pickle'):# and not (f.startswith('UF (k') or f.startswith('UF (U')): # check data and experiment
        ax = plot(ax, data, colors[data['algo']], score=score)
        # if f.startswith(f'UF (k'): # Varying kappa have specific colors
        #     ax = plot(ax, data, kappa_colors.pop(), mse=mse)
        # else:
        #     ax = plot(ax, data, colors.pop(), mse=mse)

plt.xlabel('N training samples')
if score == 'mse':
    plt.ylabel('Mean squared probability error')
elif score == 'diff':
    plt.ylabel('Mean probability difference')
elif score == 'accuracy':
    plt.ylabel('Mean 0-1 Error')
elif score == 'brier':
    plt.ylabel('Brier Score')
ax.set_xscale('log')

# Sort legend
handles, labels = ax.get_legend_handles_labels()
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'./figures/{score}_{epsilon}-{d}_std{tag}-0.5split.pdf')
