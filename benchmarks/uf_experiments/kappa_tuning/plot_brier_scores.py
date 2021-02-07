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

n_gaussians = 3
oblique = False
if oblique:
    tag = '_oblique'
else:
    tag = ''

data_dir = f'./{n_gaussians}gaussians{tag}_results/'

n = 100
d = 20

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
        score = data['y_probs']
        score[np.arange(len(score)), data['y'].astype(int)] -= 1
        score = np.mean(np.sum(score**2, axis=1))
        # score = brier_score_loss(data['y'], data['y_probs'][:, 1])
        score_dict[f'{data["kappa"]}'][data["epsilon"]].append(score)

f, ax = plt.subplots(1, 1, figsize=(5, 5))
kappas = list(score_dict.keys())
kappas = np.sort(kappas)
for c, kappa in zip(kappa_colors, kappas):
# for kappa in kappas:  
    xy_dict = score_dict[kappa]
    epsilons = list(xy_dict.keys())
    epsilons = np.sort(epsilons)
    means = np.asarray([np.mean(xy_dict[epsilon]) for epsilon in epsilons])
    stds = np.asarray([np.std(xy_dict[epsilon]) for epsilon in epsilons])
    # plt.errorbar(epsilons, means, yerr=stds, label=f'k={kappa}', c=c)
    plt.plot(epsilons, means, label=f'k={kappa}', c=c)
    # plt.fill_between(epsilons, means-stds, means+stds, color=c, alpha=0.25)

for c, name in zip(colors, ['RF', 'IRF', 'SigRF', 'UF (Uncorrected)']):
    xy_dict = defaultdict(list)
    for f in os.listdir(data_dir):
        if f.startswith(f'{name}_{n}x{d}'):
            data = unpickle(data_dir + f)
            score = data['y_probs']
            score[np.arange(len(score)), data['y'].astype(int)] -= 1
            score = np.mean(np.sum(score**2, axis=1))
            # score = brier_score_loss(data['y'], data['y_probs'][:, 1])
            xy_dict[data["epsilon"]].append(score)

    epsilons = list(xy_dict.keys())
    epsilons = np.sort(epsilons)
    means = np.asarray([np.mean(xy_dict[epsilon]) for epsilon in epsilons])
    stds = np.asarray([np.std(xy_dict[epsilon]) for epsilon in epsilons])
    plt.plot(epsilons, means, label=f'{name}', c=c)


plt.xlabel('Distance between means')
plt.ylabel('Brier score')
plt.legend()
plt.tight_layout()
plt.savefig(f'./figures/brier_scores_{n_gaussians}gaussians{tag}_{n}x{d}.pdf')
