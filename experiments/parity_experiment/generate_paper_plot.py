#%%
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns 
import matplotlib
import numpy as np
import pickle
from proglearn.sims import generate_gaussian_parity

from sklearn.model_selection import StratifiedKFold
from math import log2, ceil 


#%%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

#%%#%% Plotting the result
#mc_rep = 50
fontsize=30
labelsize=28


fig = plt.figure(constrained_layout=True,figsize=(21,30))
gs = fig.add_gridspec(30, 21)

colors = sns.color_palette('Dark2', n_colors=2)

X, Y = generate_gaussian_parity(750)
Z, W = generate_gaussian_parity(750, angle_params=np.pi/2)
P, Q = generate_gaussian_parity(750, angle_params=np.pi/4)

ax = fig.add_subplot(gs[:6,:6])
ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian XOR', fontsize=30)

plt.tight_layout()
ax.axis('off')
#plt.savefig('./result/figs/gaussian-xor.pdf')

#####################
ax = fig.add_subplot(gs[:6,7:13])
ax.scatter(Z[:, 0], Z[:, 1], c=get_colors(colors, W), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian N-XOR', fontsize=30)
ax.axis('off')

#####################
ax = fig.add_subplot(gs[:6,14:20])
ax.scatter(P[:, 0], P[:, 1], c=get_colors(colors, Q), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian R-XOR', fontsize=30)
ax.axis('off')

######################
mean_error = unpickle('plots/mean_xor_nxor.pickle')

n_xor = (100*np.arange(0.5, 7.25, step=0.25)).astype(int)
n_nxor = (100*np.arange(0.5, 7.50, step=0.25)).astype(int)

n1s = n_xor
n2s = n_nxor

ns = np.concatenate((n1s, n2s + n1s[-1]))
ls=['-', '--']
algorithms = ['XOR Forest', 'N-XOR Forest', 'Lifelong Forest', 'Naive Forest']


TASK1='XOR'
TASK2='N-XOR'

fontsize=30
labelsize=28

colors = sns.color_palette("Set1", n_colors = 2)

ax1 = fig.add_subplot(gs[7:13,2:9])
# for i, algo in enumerate(algorithms):
ax1.plot(n1s, mean_error[0,:len(n1s)], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[1], c=colors[1], ls=ls[1], lw=3)

ax1.plot(ns, mean_error[1], label=algorithms[2], c=colors[0], ls=ls[np.sum(1 > 1).astype(int)], lw=3)

ax1.plot(ns, mean_error[4], label=algorithms[3], c='g', ls=ls[np.sum(1 > 1).astype(int)], lw=3)

ax1.set_ylabel('Generalization Error (%s)'%(TASK1), fontsize=fontsize)
ax1.legend(loc='upper left', fontsize=20, frameon=False)
ax1.set_ylim(0.04, 0.21)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0.05,0.1,0.15, 0.2])
ax1.set_xticks([250,750,1500])
#ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
ax1.set_title('XOR', fontsize=30)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

#######################
mean_error = unpickle('plots/mean_xor_nxor.pickle')

algorithms = ['XOR Forest', 'N-XOR Forest', 'Lifelong Forest', 'Naive Forest']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[7:13,12:19])
ax1.plot(n1s, mean_error[0,:len(n1s)], label=algorithms[0], c=colors[1], ls=ls[np.sum(0 > 1).astype(int)], lw=3)
ax1.plot(ns[len(n1s):], mean_error[2, len(n1s):], label=algorithms[1], c=colors[1], ls=ls[1], lw=3)

ax1.plot(ns[len(n1s):], mean_error[3, len(n1s):], label=algorithms[2], c=colors[0], ls=ls[1], lw=3)
ax1.plot(ns[len(n1s):], mean_error[5, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax1.set_ylabel('Generalization Error (%s)'%(TASK2), fontsize=fontsize)
#ax1.legend(loc='upper left', fontsize=18, frameon=False)
#         ax1.set_ylim(-0.01, 0.22)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_ylim(0.04, 0.21)
ax1.set_yticks([0.05,0.1,0.15, 0.2])
# ax1.set_yticks([0.15, 0.25, 0.35])
#ax1.set_yticks([0.15, 0.2])
ax1.set_xticks([250,750,1500])
#ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")

#ax1.set_ylim(0.11, 0.21)

right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)

# ax1.set_ylim(0.14, 0.36)
ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

ax1.set_title('N-XOR', fontsize=30)

##################
mean_te = unpickle('plots/mean_te_xor_nxor.pickle')
algorithms = ['Lifelong BTE', 'Lifelong FTE', 'Naive BTE', 'Naive FTE']

TASK1='XOR'
TASK2='N-XOR'

ax1 = fig.add_subplot(gs[15:21,2:9])

ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)

ax1.plot(ns[len(n1s):], mean_te[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)

ax1.plot(ns, mean_te[2], label=algorithms[2], c='g', ls=ls[0], lw=3)
ax1.plot(ns[len(n1s):], mean_te[3, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax1.set_ylabel('Forward/Backward \n Transfer Efficiency (FTE/BTE)', fontsize=fontsize)
ax1.legend(loc='upper left', fontsize=20, frameon=False)
#ax1.set_ylim(.99, 1.4)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([.5,1,1.5,2,2.5])
ax1.set_ylim(.015, 2.55)
ax1.set_xticks([250,750,1500])
ax1.set_title('XOR N-XOR experiment', fontsize=30)
#ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

######################
mean_te = unpickle('plots/mean_te_xor_rxor.pickle')
algorithms = ['Lifelong BTE', 'Lifelong FTE', 'Naive BTE', 'Naive FTE']

TASK1='XOR'
TASK2='R-XOR'

ax1 = fig.add_subplot(gs[15:21,12:19])

ax1.plot(ns, mean_te[0], label=algorithms[0], c=colors[0], ls=ls[0], lw=3)

ax1.plot(ns[len(n1s):], mean_te[1, len(n1s):], label=algorithms[1], c=colors[0], ls=ls[1], lw=3)

ax1.plot(ns, mean_te[2], label=algorithms[2], c='g', ls=ls[0], lw=3)
ax1.plot(ns[len(n1s):], mean_te[3, len(n1s):], label=algorithms[3], c='g', ls=ls[1], lw=3)

ax1.set_ylabel('Forward/Backward \n Transfer Efficiency (FTE/BTE)', fontsize=fontsize)
#ax1.legend(loc='upper left', fontsize=20, frameon=False)
#ax1.set_ylim(.99, 1.4)
ax1.set_xlabel('Total Sample Size', fontsize=fontsize)
ax1.tick_params(labelsize=labelsize)
ax1.set_yticks([0,.5,1])
ax1.set_ylim(.25, 1.24)
ax1.set_xticks([250,750,1500])
ax1.set_title('XOR R-XOR experiment', fontsize=30)
#ax1.axvline(x=750, c='gray', linewidth=1.5, linestyle="dashed")
right_side = ax1.spines["right"]
right_side.set_visible(False)
top_side = ax1.spines["top"]
top_side.set_visible(False)
ax1.hlines(1, 50,1500, colors='gray', linestyles='dashed',linewidth=1.5)

ax1.text(400, np.mean(ax1.get_ylim()), "%s"%(TASK1), fontsize=26)
ax1.text(900, np.mean(ax1.get_ylim()), "%s"%(TASK2), fontsize=26)

########################################################
ax = fig.add_subplot(gs[23:29,2:9])
with open('plots/mean_angle_te.pickle','rb') as f:
    te = pickle.load(f)
angle_sweep = range(0,90,1)

ax.plot(angle_sweep,te,c='r',linewidth = 3)
ax.set_xticks(range(0,91,15))
ax.tick_params(labelsize=labelsize)
ax.set_xlabel('Angle of Rotation (Degrees)', fontsize=fontsize)
ax.set_ylabel('Backward Transfer Efficiency (XOR)', fontsize=fontsize)
#ax.set_title("XOR vs. Rotated-XOR", fontsize = fontsize)
ax.hlines(1,0,90, colors='grey', linestyles='dashed',linewidth=1.5)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

#####################################
ax = fig.add_subplot(gs[23:29,12:19])

with open('plots/mean_sample_te.pickle','rb') as f:
    te = pickle.load(f)
task2_sample_sweep = (2**np.arange(np.log2(60), np.log2(5010)+1, .25)).astype('int')

ax.plot(task2_sample_sweep,te,c='r',linewidth = 3)
ax.hlines(1, 60,5200, colors='gray', linestyles='dashed',linewidth=1.5)
ax.set_xscale('log')
ax.set_xticks([])
ax.set_yticks([0.98,1,1.02,1.04])
ax.tick_params(labelsize=26)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.text(50, np.mean(ax.get_ylim())-.042, "50", fontsize=labelsize)
ax.text(500, np.mean(ax.get_ylim())-.042, "500", fontsize=labelsize)
ax.text(5000, np.mean(ax.get_ylim())-.042, "5000", fontsize=labelsize)

ax.text(50, np.mean(ax.get_ylim())-.047, "Number of $25^\circ$-RXOR Training Samples", fontsize=fontsize-3)
ax.set_ylabel('Backward Transfer Efficiency (XOR)',fontsize=fontsize)

right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)


plt.savefig('./plots/parity_exp.pdf')
# %%
