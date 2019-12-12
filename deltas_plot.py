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

runs = ['thresh-2']

fig, ax = plt.subplots(len(runs), 1)
fig.set_size_inches(5, 5*len(runs))
for i, run in enumerate(runs):
    print(run)
    p = ax if len(runs) ==1 else ax[i]
    with open(run+'/deltas.pkl', 'rb') as f:
        deltas = pkl.load(f)
    colors = cm.rainbow(np.linspace(1, 0, len(deltas)//2+10))
    ind = 0
    for dl in tqdm(deltas):
        if len(dl) > 0:
        	p.plot(list(np.log(dl)), color=colors[ind])
        	ind += 0
        # for d in dl:
        #     p.plot(np.log(d), color=next(colors), marker='-')
plt.savefig(run+'/deltas.png')
plt.xlabel("Iteration of Neural Dynamics Sim")
plt.ylabel("Log of ||delta u||/||u||")
plt.show()

