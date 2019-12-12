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

def create_dir(directory):
    """Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, default='test', help="Run name")
parser.add_argument('--tanh_factors', type=str, default= "1", 
  help='tanh parameter g: tanh(gx), one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--distance_parameter', type=str, 
  help='distance parameter to define the radius for the feedforward connection, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--stride', type=str, default= "2", 
  help='stride for the feedforward connection, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--gamma_factor', type=float, default=0.0, 
  help='feedback parameter')
parser.add_argument('--mult_factor', type=str, default= "1", 
   help='multiplication factor, one per layer, seperated by comma', metavar='a,b,c')
parser.add_argument('--NpSs', type=str, default= "4",
  help='neuron per site, one per layer, seperated by comma', metavar='a,b,c')

parser.add_argument('--batch_size', type=int, default=1, help='batch size during training')
parser.add_argument('--epoch_size', type=int, default=1000, help='random sample to take for each "epoch"')
parser.add_argument('--max_iters', type=int, default=3000, help='How many iterations (at most) to run the neural dynamics')
parser.add_argument('--threshold', type=float, default=0.0001, help='Threshold on delta for convergence')
parser.add_argument('--bleed', type=int, default=0, help='Whether or not to carry state through iterations')
parser.add_argument('--samples_rs', type=int, default=0, help="Random seed for sample ordering")

# parser.add_argument('--min_iters', type=int, default=100, help='How many iterations (at least) to run the neural dynamics')

# parser.add_argument('--ordered')

args = parser.parse_args()

# if args.min_iters > args.max_iters:
#     args.min_iters = args.max_iters

create_dir(args.run_name)
summary_dir = args.run_name+'/logs'
rep_dir = args.run_name+'/reps'
checkpoint_dir = args.run_name+'/checkpoints'
create_dir(summary_dir)
create_dir(rep_dir)
create_dir(checkpoint_dir)
summary_writer = SummaryWriter(log_dir=summary_dir)

tanh_factors = list(map(float, args.tanh_factors.split(',')))
distance_parameter= list(map(float, args.distance_parameter.split(',')))
stride= list(map(float, args.stride.split(',')))
gamma_factor= float(args.gamma_factor)
mult_factor  = list(map(float, args.mult_factor.split(',')))
NpSs = list(map(int, args.NpSs.split(',')))

lateral_distance = [0]*len(distance_parameter)

x_train, x_test, y_train, y_test = get_mnist()

# WE ASSUME FLATTENED IMAGES

image_dim = 28
channels = 1
strides = stride
distances = distance_parameter
distances_lateral = lateral_distance
tanh_factors = tanh_factors
layers = len(distance_parameter)
gamma = gamma_factor
mult_factors = mult_factor

network = deep_network_GPU(image_dim = image_dim, channels = channels, 
                           NpSs=NpSs, strides=strides, distances=distances, 
                           layers=layers, gamma=gamma, lr=5e-3, lr_floor = 1e-4, 
                           decay=0.5, distances_lateral = distances_lateral, 
                           tanh_factors = tanh_factors, mult_factors = mult_factors, euler_step=0.2)

print('NpS: ', network.NpSs)
print('Strides: ', network.strides)
print('Distances: ', network.distances)
print('Lateral Distances: ', network.lateral_distances)
print('gamma :', network.gamma)
print('tanh_factor: ', network.tanh_factors)
print('Dimensions: ', network.dimensions)

np.random.seed(args.samples_rs)
for i in range(120):
    if i%5 == 0:
        with open(args.run_name + '/update_distr.pkl', 'wb') as f:
            pickle.dump(network.update_distr, f)
        with open(args.run_name + '/deltas.pkl', 'wb') as f:
            pickle.dump(network.max_deltas, f)
    if i%30 == 0:
        with open(checkpoint_dir + '/network_g_val{0}_m{1}_g{2}_r{3}_NpS{4}_full--iter{5}.pkl'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs, i), 'wb') as output:
            pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)
    indices = np.arange(0, x_train.shape[0])
    rand_indices = np.random.choice(indices, size=1000)
    # print(rand_indices)
    # print(x_train.shape)
    x_train_rand = x_train[rand_indices]
    conversions = network.training(epochs=1, images=x_train_rand, batch_size=args.batch_size, max_iters=args.max_iters, threshold=args.threshold, bleed=(args.bleed==1))
    summary_writer.add_scalar('training/converge_pct', conversions[-1], i)
    # raise NotImplementedError
    if (i+1)%15 == 0 or i == 0:
        print("Checking accuracy of LinearSVC, 'Epoch':", i)
        # train_representations = np.zeros((x_train.shape[0], np.sum(network.dimensions)))
        print("Train reps")
        tic = time.time()
        train_representations, _ = network.neural_dynamics(x_train, max_iters=args.max_iters, threshold=args.threshold, verbose=True)
        print("Parallelized network sim took:", time.time()-tic)
        train_representations = train_representations.T
        np.save(rep_dir+'/train_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}--iter{5}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs, i+1), train_representations)
        # print(train_representations.shape)
        # for i in tqdm(range(x_train.shape[0])):
        #     train_rep, _ = network.neural_dynamics(x_train[i:i+1], max_iters=10)
        #     train_representations[i] = train_rep[:,0]
            # print("Is it working?", np.allclose(train_representations[i], train_representations[i]))
            # print(np.linalg.norm(train_representations[i] - train_representations[i]))
        print("Train reps")
        tic = time.time()
        test_representations, _ = network.neural_dynamics(x_test, max_iters=args.max_iters, threshold=args.threshold, verbose=True)
        print("Parallelized network sim took:", time.time()-tic)
        test_representations = test_representations.T
        np.save(rep_dir+'/test_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}--iter{5}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs, i+1), test_representations)

        clf_output = LinearSVC(tol=1e-4, dual=False, class_weight='balanced')
        print("Fitting LinearSVC")
        tic = time.time()
        print(train_representations[:,-network.dimensions[-1]:].shape)
        clf_output.fit(train_representations[:,-network.dimensions[-1]:], y_train)
        print("Fitting took:", time.time()-tic)
        train_score = clf_output.score(train_representations[:, -network.dimensions[-1]:], y_train)
        test_score = clf_output.score(test_representations[:, -network.dimensions[-1]:], y_test)

        print('Test Score: ', test_score)
        print('Train Score: ', train_score, flush=True)
        summary_writer.add_scalar('training/score', train_score, i)
        summary_writer.add_scalar('testing/score', test_score, i)

with open(checkpoint_dir + '/network_g_val{0}_m{1}_g{2}_r{3}_NpS{4}_full--final.pkl'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), 'wb') as output:
    pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)

#train_representations_output = np.zeros((x_train.shape[0], network.dimensions[-1])
# train_representations = np.zeros((x_train.shape[0], np.sum(network.dimensions)))
# for i in range(x_train.shape[0]):
#     train_rep, _ = network.neural_dynamics(x_train[i])
#     train_representations[i] = train_rep
#     if (i+1)%1000==0:
#         print(i+1)

# np.save('train_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), train_representations)

# test_representations = np.zeros((x_test.shape[0], np.sum(network.dimensions)))
# for i in range(x_test.shape[0]):
#     test_rep, _ = network.neural_dynamics(x_test[i])
#     test_representations[i] = test_rep
#     if (i+1)%1000==0:
#         print(i+1)

# np.save('test_representations_g_val{0}_m{1}_g{2}_r{3}_NpS{4}.npy'.format(tanh_factors, mult_factor, gamma_factor, distance_parameter, NpSs), test_representations)

# clf_output = LinearSVC(random_state=0, tol=1e-5, max_iter = 1e6, class_weight='balanced')
# clf_output.fit(train_representations[:,-network.dimensions[-1]:], y_train)
# train_score = clf_output.score(train_representations[:, -network.dimensions[-1]:], y_train)
# test_score = clf_output.score(test_representations[:, -network.dimensions[-1]:], y_test)

# print('Test Score: ', test_score)
# print('Train Score: ', train_score)
