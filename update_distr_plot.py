import numpy as np
import pickle as pkl
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.svm import LinearSVC
from mnist_data import get_mnist
import argparse
from tensorboardX import SummaryWriter

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

runs = ['default', 'thresh-2']

# fig, ax = plt.subplots(len(runs), 1)
# fig.set_size_inches(5, 5*len(runs))
for i, run in enumerate(runs):
    print(run)
    # p = ax if len(runs) ==1 else ax[i]
    with open(run+'/update_distr.pkl', 'rb') as f:
        update_distr = pkl.load(f)
    # colors = cm.rainbow(np.linspace(1, 0, len(deltas)*1000+30))
    all_updates = np.zeros(len(update_distr)*1000)
    print(update_distr)
    ind = 0
    for updates in update_distr:
        to_add = min(len(updates), 1000)
        all_updates[ind:ind+to_add] = updates[:to_add]
        ind += to_add
    plt.plot(all_updates, label=run)
plt.legend()
plt.ylabel("Number of updates to converge")
plt.xlabel("Iteration #")
plt.show()
    # for i in range(5):
    #     print(deltas[i])
        # for d in dl:
        #     p.plot(np.log(d), color=next(colors), marker='-')
# plt.savefig(run+'/deltas.png')
# plt.show()

