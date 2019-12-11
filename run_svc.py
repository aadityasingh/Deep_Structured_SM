import numpy as np
import time
import pickle
from snn_multipleneurons_fast import *
import sys
import os
from sklearn.svm import LinearSVC
from mnist_data import get_mnist
import argparse
from tensorboardX import SummaryWriter

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

run = input("Run folder: ")

x_train, x_test, y_train, y_test = get_mnist()

clf_output = LinearSVC(tol=1e-6, dual=False, class_weight='balanced')

if run == 'raw-mnist':
  print("Doing MNIST")
  tic = time.time()
  clf_output.fit(x_train, y_train)
  train_score = clf_output.score(x_train, y_train)
  test_score = clf_output.score(x_test, y_test)
  print("Time elapsed", time.time()-tic)

  print("MNIST baseline:")
  print('Test Score: ', test_score)
  print('Train Score: ', train_score, flush=True)
else:
  train_representations = np.load(run+'/reps/train_representations_final.npy')
  test_representations = np.load(run+'/reps/test_representations_final.npy')
  print(train_representations.shape, test_representations.shape)
  print("Read in files, doing SVC")
  tic = time.time()
  clf_output.fit(train_representations[:, -768:], y_train)
  train_score = clf_output.score(train_representations[:, -768:], y_train)
  test_score = clf_output.score(test_representations[:, -768:], y_test)
  print("Time elapsed", time.time()-tic)

  print("Run: {}".format(run))
  print('Test Score: ', test_score)
  print('Train Score: ', train_score, flush=True)
  # clf_output2 = LinearSVC(tol=1e-5, dual=False, class_weight='balanced', verbose=True)


# MNIST baseline:
# Test Score:  0.9155
# Train Score:  0.9269166666666667
