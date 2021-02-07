import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from collections import defaultdict
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve


def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data

n_gaussians = 3
oblique = False
if oblique:
    tag = '_oblique'
else:
    tag = ''

data_dir = f'./{n_gaussians}gaussians{tag}_results/'

n = 100
d = 2

kappa_colors = [
    '#deebf7',
    '#c6dbef',
    '#9ecae1',
    '#6baed6',
    '#4292c6',
    '#2171b5',
    '#08519c',
]

colors = [
    '#fdbf6f',
    '#4daf4a',
    '#ff7f00',
    '#a65628',
]

score_dict = defaultdict(lambda: defaultdict(list))
for f in os.listdir(data_dir):
    if f.startswith(f'UF_{n}x{d}'):
        data = unpickle(data_dir + f)
        score_dict[data["epsilon"]][f'{data["kappa"]}'].append((data['y'], data['y_probs'][:, 1]))

nok_dict = defaultdict(lambda: defaultdict(list))
for c, name in zip(colors, ['RF', 'IRF', 'SigRF', 'UF (Uncorrected)']):
    for f in os.listdir(data_dir):
        if f.startswith(f'{name}_{n}x{d}'):
            data = unpickle(data_dir + f)
            nok_dict[data["epsilon"]][name].append((data['y'], data['y_probs'][:, 1]))

f, axes = plt.subplots(2, 6, figsize=(15, 5), sharex=True, sharey=True)

epsilons = np.sort(list(score_dict.keys()))
for i, epsilon in enumerate(epsilons):
    ax = axes[i // 6, i % 6]
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    kappas = np.sort(list(score_dict[epsilon].keys()))    
    for c, kappa in zip(kappa_colors, kappas):
        y, y_pos_prob = list(zip(*score_dict[epsilon][kappa]))
        y = np.hstack(y)
        y_pos_prob = np.hstack(y_pos_prob)
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, y_pos_prob, n_bins=10, strategy='uniform')
        ax.plot(mean_predicted_value, fraction_of_positives, label=f'k={kappa}', c=c)

    for c, name in zip(colors, nok_dict[epsilon].keys()):
        y, y_pos_prob = list(zip(*nok_dict[epsilon][name]))
        y = np.hstack(y)
        y_pos_prob = np.hstack(y_pos_prob)
        fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, y_pos_prob, n_bins=10, strategy='uniform')
        ax.plot(mean_predicted_value, fraction_of_positives, label=name, c=c)
    
    ax.set_title(f'Epsilon {epsilon:.2f}')

plt.setp(axes[-1, :], xlabel='Predicted Prob')
plt.setp(axes[:, 0], ylabel='Correctly classified fraction')

axes[0, -1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'./figures/calibration_curves_{n_gaussians}gaussians{tag}_{n}x{d}.pdf')
