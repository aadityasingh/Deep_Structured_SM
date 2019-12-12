import numpy as np
import time
from sklearn.utils import shuffle
from scipy.sparse import csr_matrix, identity
from tqdm import tqdm

# if you want to use cupy for gpu 
# import cupy as cp
# from cupyx.scipy.sparse import csr_matrix as csr_gpu

# if you just wnat to use numpy and scipy for cpu
import numpy as cp
from scipy.sparse import csr_matrix as csr_gpu

np.random.seed(10)

class network_weights(object):
    def __init__(self, NpS, previous_NpS, distance_parameter, input_dim, stride, lateral_distance):
        self.distance_parameter = distance_parameter
        self.input_dim = input_dim
        self.stride = stride
        self.NpS = NpS
        self.output_dim = int(self.input_dim/self.stride)
        self.W = None
        self.L = None
        self.W_structure = None
        self.L_structure = None
        self.previous_NpS = previous_NpS
        self.lateral_distance = lateral_distance
        
    def create_h_distances(self):
        distances = np.zeros((self.output_dim**2, self.input_dim, self.input_dim))

        # Not actually used... maybe some legacy code?
        dict_input_2_position = {}
        for row_index in range(self.input_dim):
            for column_index in range(self.input_dim):
                input_index = row_index*self.input_dim + column_index
                dict_input_2_position[row_index, column_index] = input_index
                
        # dict_ooutput_2_position code is repeated
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center

                for k in range(self.input_dim):
                    for l in range(self.input_dim):
                        distances[neuron_index, k,l] = np.linalg.norm(np.array([k+0.5,l+0.5])-neuron_center)
        above_threshold = distances > self.distance_parameter
        below_threshold = distances <= self.distance_parameter
        distances[above_threshold] = 0
        distances[below_threshold] = 1
        distances = distances.reshape((self.output_dim**2, self.input_dim**2))
        return distances # Thresholded distances from input to output showing connectivity... circular not square
    
    def create_ah_distances(self):
        # dict_ooutput_2_position code is repeated
        centers = []
        dict_output_2_position = {}
        for i in range(self.output_dim):
            for j in range(self.output_dim):
                stride_padding = self.stride/2
                neuron_center = np.array([i*self.stride + stride_padding, j*self.stride + stride_padding])
                centers.append(neuron_center)
                neuron_index = i*self.output_dim + j
                dict_output_2_position[neuron_index] = neuron_center

        distances_ah = np.zeros((self.output_dim**2, self.output_dim**2))
        for row_index in list(dict_output_2_position.keys()):
            center = dict_output_2_position[row_index]
            for column_index in list(dict_output_2_position.keys()):
                other_center = dict_output_2_position[column_index]
                distances_ah[row_index, column_index] = np.linalg.norm(other_center - center)
        above_threshold = distances_ah > self.lateral_distance #*self.anti_hebbian_binary
        below_threshold = distances_ah <= self.lateral_distance #*self.anti_hebbian_binary
        distances_ah[above_threshold] = 0
        distances_ah[below_threshold] = 1
        return distances_ah
    
    def create_L(self):
        mat = self.create_ah_distances()
        print("L pre", mat.shape)
        blocks = [[mat]*self.NpS]*self.NpS
        L_mat = np.block(blocks)
        print("L shape", L_mat.shape)
        return L_mat
    
    def create_W(self):
        mat = self.create_h_distances()
        print("W pre", mat.shape)
        blocks = [[mat]*self.previous_NpS]*self.NpS
        W_mat = np.block(blocks)
        print("W shape", W_mat.shape)
        return W_mat
    
    def create_weights_matrix(self):
        self.W_structure = self.create_W()
        self.L_structure = self.create_L()
        factor = np.sqrt( ( (np.sum(self.W_structure)/self.NpS) /self.output_dim**2)) 
        self.W = self.W_structure*np.random.normal(0, 1, (self.W_structure.shape))/factor
        self.L = self.L_structure*np.identity(self.NpS * self.output_dim**2)

class deep_network_GPU(object):
    def __init__(self, image_dim, channels, NpSs, strides, distances, 
                 layers, gamma, lr, lr_floor, decay, distances_lateral, tanh_factors, mult_factors, euler_step):
        self.image_dim = image_dim
        self.channels = channels
        self.NpSs = NpSs
        self.strides = strides
        self.distances = distances
        self.lateral_distances = distances_lateral
        self.layers = layers
        self.gamma = gamma
        self.lr = lr
        self.lr_floor = lr_floor
        self.current_lr = None
        self.decay = decay
        self.conversion_tickers = []
        self.costs = []
        self.epoch=0
        self.structure = None
        self.deep_matrix_weights = None
        self.deep_matrix_structure = None
        self.deep_matrix_identity = None
        self.weights_adjustment_matrix = None
        self.weights_update_matrix = None
        self.grad_matrix = None
        self.n_images = None
        self.dict_weights = {}
        self.dimensions = []
        self.g_vec = None
        self.mult_vec = None
        self.euler_step = euler_step
        self.tanh_factors = tanh_factors # UNUSED
        self.mult_factors = mult_factors
        self.W_gpu = None
        deep_network_GPU.create_deep_network(self)
        self.update_distr = []
        self.max_deltas = []
        print("Created deep network")
    
    def create_deep_network(self):
        for i in range(self.layers+1):
            dim = int(np.prod(self.strides[:i]))
            self.dimensions.append(int((self.image_dim/dim)**2)*([self.channels]+self.NpSs)[i])
        
        for i in range(self.layers):
            layer_input_dim = int(self.image_dim/np.prod(self.strides[:i]))
            self.dict_weights[i]=network_weights(NpS=([self.channels]+self.NpSs)[i+1], distance_parameter=self.distances[i], 
                                                input_dim=layer_input_dim,
                                                stride = self.strides[i], previous_NpS = ([self.channels]+self.NpSs)[i], lateral_distance=self.lateral_distances[i])
            self.dict_weights[i].create_weights_matrix()
        
        matrix_block = []
        structure_block = []
        matrix_identity = []
        weight_adjustment_block = []
        gradient_update_block = []
        abs_structure_block = []

        for i, ele_row in enumerate(self.dimensions):
            row_block = []
            struc_block = []
            row_identity_block = []
            weights_adj_block = []
            grad_update_block = []
            abs_struc_block = []

            start_block = max(i-1, 0)
            end_block = max(len(self.dimensions)-start_block-3, 0)

            if i == 0:
                row_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                row_identity_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                weights_adj_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                grad_update_block.append(np.zeros((ele_row, np.sum(self.dimensions))))
                abs_struc_block.append(np.zeros((ele_row, np.sum(self.dimensions))))

            elif i < len(self.dimensions)-1:
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                row_block.append(self.dict_weights[i].W.T) # Feedback weights
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)
                struc_block.append(self.gamma*self.mult_factors[i]*self.dict_weights[i].W_structure.T)

                abs_struc_block.append(self.dict_weights[i-1].W_structure)
                abs_struc_block.append(self.dict_weights[i-1].L_structure)
                abs_struc_block.append(self.dict_weights[i].W_structure.T)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                row_identity_block.append(np.zeros((self.dict_weights[i].W_structure.T.shape)))

                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure/(1+self.gamma))
                weights_adj_block.append(self.dict_weights[i].W_structure.T)

                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)
                grad_update_block.append(self.dict_weights[i].W_structure.T)

                if end_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[-end_block:])))))

            elif i+1 == len(self.dimensions):
                if start_block > 0:
                    row_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    row_identity_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    weights_adj_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    grad_update_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))
                    abs_struc_block.append(np.zeros((ele_row, int(np.sum(self.dimensions[:start_block])))))

                row_block.append(self.dict_weights[i-1].W)
                row_block.append(self.dict_weights[i-1].L)
                
                struc_block.append(self.dict_weights[i-1].W_structure/self.mult_factors[i-1])
                struc_block.append(-self.dict_weights[i-1].L_structure)

                abs_struc_block.append(self.dict_weights[i-1].W_structure)
                abs_struc_block.append(self.dict_weights[i-1].L_structure)
                
                row_identity_block.append(np.zeros((self.dict_weights[i-1].W_structure.shape)))
                row_identity_block.append(np.identity(self.dict_weights[i-1].L_structure.shape[0]))
                
                weights_adj_block.append(self.dict_weights[i-1].W_structure)
                weights_adj_block.append(self.dict_weights[i-1].L_structure)
                
                grad_update_block.append(self.dict_weights[i-1].W_structure)
                grad_update_block.append(self.dict_weights[i-1].L_structure/2)

            matrix_block.append(row_block)
            structure_block.append(struc_block)
            matrix_identity.append(row_identity_block)
            weight_adjustment_block.append(weights_adj_block)
            gradient_update_block.append(grad_update_block)
            abs_structure_block.append(abs_struc_block)

        self.deep_matrix_weights = cp.asarray(np.block(matrix_block))
        self.deep_matrix_structure = cp.asarray(np.block(structure_block))
        self.deep_matrix_identity = cp.asarray(np.block(matrix_identity))
        self.weights_adjustment_matrix = cp.asarray(np.block(weight_adjustment_block))
        self.weights_update_matrix = cp.asarray(np.block(gradient_update_block))
        self.structure = cp.asarray(np.block(abs_structure_block))
    
    def activation_function(self, vec):
        return cp.tanh(vec)
    
    def neural_dynamics(self, imgs, max_iters=3000, threshold=1e-4, r_init = None, verbose=False):
        conversion_ticker = 0
        # x = imgs
        u_vecs = cp.asarray(np.zeros((np.sum(self.dimensions), imgs.shape[0])))
        r_vecs = np.zeros((np.sum(self.dimensions), imgs.shape[0]))
        r_vecs[:self.channels*self.image_dim**2] = imgs.T
        if r_init is not None:
            # print("SHOULND'T PRINT THIS")
            r_vecs[self.channels*self.image_dim**2:] = r_init[self.channels*self.image_dim**2:]
        r_vecs = cp.asarray(r_vecs)
        # delta = [cp.inf]*self.layers
        delta = cp.inf
        deltas = []
        self.W_gpu = csr_gpu(self.deep_matrix_weights*self.deep_matrix_structure + self.deep_matrix_identity)
        # updates = 0
        for updates in tqdm(iterable=range(max_iters), disable=(not verbose)):
            if delta < threshold:#all(ele < 1e-4 for ele in delta):
                conversion_ticker=1
                if verbose:
                    print("Iteration converged", updates)
                break
            lr = max((self.euler_step/(1+0.005*updates)), 0.05)
            # print(self.W_gpu.shape)
            # print(r_vec.shape)
            delta_us = -u_vecs + self.W_gpu.dot(r_vecs)
            u_vecs[self.channels*self.image_dim**2:] += lr*delta_us[self.channels*self.image_dim**2:]
            r_vecs[self.channels*self.image_dim**2:] = self.activation_function(u_vecs[self.channels*self.image_dim**2:])
            # updates += 1
            # if (updates+1)%min_iters == 0:
            #     for layer in range(1, self.layers+1):
            #         start_token_large = np.sum(self.dimensions[:layer])
            #         end_token_large = np.sum(self.dimensions[:layer+1])
            #         start_token_small = int(np.sum(self.dimensions[1:][:layer-1]))
            #         end_token_small = np.sum(self.dimensions[1:][:layer])
            #         delta_layer = cp.linalg.norm(delta_us[start_token_small:end_token_small])/cp.linalg.norm(u_vecs[start_token_large:end_token_large])
            #         delta[layer-1] = delta_layer
            delta = cp.linalg.norm(delta_us[self.channels*self.image_dim**2:])/cp.linalg.norm(u_vecs[self.channels*self.image_dim**2:])
            deltas.append(delta)  
            if verbose and (updates+1)%10 == 0:
                print(str(updates+1)+" delta "+ str(delta), flush=True)
        self.update_distr[-1].append(updates)
        self.max_deltas[-1].append(deltas)
        return r_vecs, conversion_ticker
    
    def update_weights(self, r_vecs):
        self.current_lr = max(self.lr/(1+self.decay*self.epoch), self.lr_floor)
        #r_vec = cp.asnumpy(r_vec)
        update_matrix = r_vecs@r_vecs.T # Sum of rank one products of r_vecs across images
        grad_weights = self.weights_update_matrix*(update_matrix - r_vecs.shape[1]*self.weights_adjustment_matrix*self.deep_matrix_weights)
        self.deep_matrix_weights += self.current_lr*grad_weights
                
    def training(self, epochs, images, batch_size=1, max_iters=3000, threshold=1e-4, bleed=False):
        print("started training")
        self.n_images = images.shape[0]
        for epoch in range(epochs):
            self.update_distr.append([])
            self.max_deltas.append([])
            img_array = shuffle(images, random_state = epoch)
            epoch_start = time.time()
            sum_ticker = 0
            rs = np.zeros((np.sum(self.dimensions), batch_size))
            for i in tqdm(range(self.n_images//batch_size)):
                # print(img_array[i:i+1].shape)
                last_r = rs if bleed else None
                rs, conversion_ticker = self.neural_dynamics(img_array[i:i+batch_size], max_iters=max_iters, threshold=threshold, r_init=last_r, verbose=False)
                # print(rs.shape)
                sum_ticker += conversion_ticker
                self.update_weights(rs)
            if self.n_images%batch_size > 0:
                rs, conversion_ticker = self.neural_dynamics(img_array[(self.n_images//batch_size):], max_iters=max_iters)
                sum_ticker += conversion_ticker
                self.update_weights(rs)
            self.epoch+=1
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            print("Conversion", float(sum_ticker)/self.n_images)
            self.conversion_tickers.append(float(sum_ticker)/self.n_images)
            print(self.deep_matrix_weights[self.deep_matrix_weights != 0].shape)
            # print(self.deep_matrix_weights.shape, (self.deep_matrix_weights[self.deep_matrix_weights != 0])[:10])
            print('Epoch: {0}\nTime_Taken: {1}\nConversion: {2}\nCurrent Learning Rate: {3}\n\n'.format(self.epoch, epoch_time, self.conversion_tickers[-1], self.current_lr))
        return self.conversion_tickers
