"""
Build a Trainable CNN from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - argmax_rows
def argmax_rows(matrix):
    # TODO: return the index of the largest element in each row of a 2D array
    n , p = matrix.shape
    val = -np.inf*np.ones(n)
    maxi = np.zeros(n, dtype = int)

    for i in range(n):
        for j in range(p):
            if matrix[i,j]>val[i]:
                val[i] = matrix[i,j]
                maxi[i] = j
    return maxi

# Step 2 - row_max
import numpy as np

def row_max(matrix):
    # TODO: return the maximum value of each row of `matrix` with keepdims True for broadcasting.
        # TODO: return the index of the largest element in each row of a 2D array
    n , p = matrix.shape
    bests=  matrix[:, 0].copy() 
   

    for i in range(n):
        for j in range(1,p):
            if matrix[i,j]>bests[i]:
                bests[i] = matrix[i,j]
                
    return bests.reshape(n,1)

# Step 3 - row_sum
import numpy as np

def row_sum(matrix):
    """Return per-row sums of a 2D array with shape (N, 1)."""
    # TODO: return the sum along axis 1 keeping the reduced dimension
    n, p = matrix.shape
    sums = np.zeros_like(matrix[:,0].copy())
    for i in range(n):
        for j in range(p):
            sums[i] += matrix[i][j]
    return sums.reshape(n,1)

# Step 4 - exp_shifted
import numpy as np

def exp_shifted(logits):
    """Subtract per-row max from logits and exponentiate elementwise."""
    # TODO: shift each row of logits by its max and return elementwise exp
    logits = logits - row_max(logits)*np.ones_like(logits)
    e = np.exp(logits)
    return e

# Step 5 - stable_softmax
def stable_softmax(logits):
    # TODO: Compute a numerically stable softmax row-wise over (N, C) logits.
    e = exp_shifted(logits)
    return e /(row_sum(e))

# Step 6 - one_hot
def one_hot(labels, num_classes):
    N = len(labels)
    # TODO: convert integer labels into a (N, num_classes) one-hot float matrix
    matrix = np.zeros((N,num_classes))
    for i in range(N):
        matrix[i][labels[i]] =1
    return matrix

# Step 7 - gather_true_class_probs
def gather_true_class_probs(probs, labels):
    # TODO: return probs[i, labels[i]] for every row i as a 1D length-N array.
    N = len(probs)
    res = np.zeros(N)
    for i in range(N):
        res[i] =  probs[i,labels[i]]
    return res

# Step 8 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(probs, labels, eps=1e-12):
    # TODO: return the mean negative log-likelihood of the true-class probabilities
    N = len(probs)
    true_probs = gather_true_class_probs(probs, labels)
    true_probs = np.clip(true_probs,eps,1.0)
    res = -np.log(true_probs).sum()
    return res/N

# Step 9 - accuracy
def accuracy(logits_or_probs, labels):
    # TODO: return the fraction of rows whose argmax matches the integer label.
    max_probs = argmax_rows(logits_or_probs)
    return (max_probs==labels).sum()/len(logits_or_probs)

# Step 10 - he_std
def he_std(fan_in):
    # TODO: return the He initialization standard deviation sqrt(2 / fan_in).
    return np.sqrt(2/fan_in)

# Step 11 - he_init
def he_init(shape, fan_in, seed):
    # TODO: sample a weight tensor from a normal distribution scaled by He std using the seed.
    np.random.seed(seed)
    return np.random.randn(*shape)*he_std(fan_in)

# Step 12 - init_zero_bias
import numpy as np

def init_zero_bias(length):
    # TODO: return a 1D float array of zeros with the given length.
    return np.zeros(length)

# Step 13 - pad_2d
def pad_2d(images, pad):
    
    
    N,C,H,W = images.shape
    out = np.zeros((N,C,H+2*pad,W+2*pad), dtype = images.dtype)
   
    out[:,:,pad:H+pad,pad:W+pad] = images
    return out

# Step 14 - output_spatial_size
def output_spatial_size(input_size, kernel, stride, padding):
    return (input_size+2*padding-kernel)//stride +1

# Step 15 - im2col
def im2col(images, kernel_h, kernel_w, stride, padding):
    # TODO: Unroll overlapping patches of a 4D image tensor into a 2D column matrix.
    
    N,C,H,W = images.shape
    out_h = output_spatial_size(H,kernel_h,stride,padding)
    out_w = output_spatial_size(W,kernel_w,stride,padding)
    x = pad_2d(images,padding)

    cols = np.zeros((N,C,kernel_h,kernel_w,out_h,out_w), dtype = x.dtype)
    for a in range(kernel_h):
        for b in range(kernel_w):
            cols[:,:,a,b] = x[:,:,a:a+out_h*stride:stride, b:b+out_w*stride:stride]
        
    return cols.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)

# Step 16 - col2im
def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    # TODO: re-roll a (N*out_h*out_w, C*kh*kw) column matrix back into a (N, C, H, W) tensor
    N,C,H,W  = input_shape
    out_h = output_spatial_size(H, kernel_h, stride, padding)
    out_w = output_spatial_size(W, kernel_w, stride, padding)
    x = np.zeros((N, C, H + 2*padding, W + 2*padding), dtype=cols.dtype)
    c6 = cols.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    for a in range(kernel_h):
        for b in range(kernel_w):
            x[:, :, a:a + stride*out_h:stride, b:b + stride*out_w:stride] += c6[:, :, a, b]
    return x[:,:,padding:padding+H, padding:padding+W]

# Step 17 - conv2d_forward
def conv2d_forward(x, weights, bias, stride, padding):
   
    F,_,kh, kw = weights.shape
    N,C,H,W = x.shape
    out_h = output_spatial_size(H, kh, stride, padding)
    out_w = output_spatial_size(W, kw, stride, padding)

    cols = im2col(x,kh,kw, stride, padding)
    w_mat = weights.reshape(F,-1)
    y = cols @ w_mat.T + bias
    y = y.reshape(N,out_h,out_w,F).transpose(0, 3, 1, 2)
    
    cache = {
        "x_shape": x.shape,
        "weights": weights,
        "cols": cols,
        "stride": stride,
        "padding": padding,
        "kernel_h": kh,
        "kernel_w": kw,
    }
    return y,cache

# Step 18 - conv2d_grad_input
def conv2d_grad_input(d_out, cache):
    # TODO: backprop d_out through the conv input using col2im
    weights = cache['weights']
    x_shape = cache['x_shape']
    padding = cache['padding']
    stride = cache['stride']
    kh = cache['kernel_h']
    kw = cache['kernel_w']
    cols = cache['cols']
    F = weights.shape[0]

    d_mat = d_out.transpose(0,2,3,1).reshape(-1,F)
    
    
    dcols = d_mat@ weights.reshape(F,-1)
    return col2im(dcols,x_shape,kh,kw,stride,padding)

# Step 19 - conv2d_grad_weights
def conv2d_grad_weights(d_out, cache):
    """return dL/dW shaped (C_out, C_in, kH, kW) from d_out and the im2col cache."""
    weights  = cache['weights']
    
    cols = cache['cols']

    F = weights.shape[0]
    d_mat = d_out.transpose(0, 2, 3, 1).reshape(-1, F)
    dW = (d_mat.T@ cols).reshape(weights.shape)
    return dW

# Step 20 - conv2d_grad_bias
def conv2d_grad_bias(d_out):
    # TODO: return a length C_out gradient by reducing d_out over batch and spatial axes
    
    
    db = d_out.sum(axis=(0,2,3))
    return db

# Step 21 - conv2d_backward
def conv2d_backward(d_out, cache):
    "return (dx, dW, db) using the conv2d gradient helpers and the forward cache"
    return conv2d_grad_input(d_out,cache), conv2d_grad_weights(d_out,cache),conv2d_grad_bias(d_out)

# Step 22 - maxpool2d_forward
def maxpool2d_forward(x, kernel, stride):
    # TODO: run 2D max pooling and cache the in-window argmax of each output cell.
    N,C,H,W = x.shape
    out_h = output_spatial_size(H,kernel,stride,0)
    out_w = output_spatial_size(W,kernel,stride,0)
    out = np.zeros((N,C,out_h,out_w), dtype =  x.dtype)
    argmax= np.zeros((N,C,out_h,out_w), dtype =  int)
    for i in range(out_h):
        for j in range(out_w):
            win = x[:,:,i*stride:i*stride+kernel,j*stride:j*stride+kernel].reshape(N,C,-1)
            out[:,:,i,j] = win.max(axis=2)
            argmax[:,:,i,j]  = win.argmax(axis=2)
    return out,{"x_shape": x.shape, "argmax": argmax,
             "kernel": kernel, "stride": stride}

# Step 23 - scatter_grad_window
import numpy as np

def scatter_grad_window(grad_value, argmax_index, kernel):
    "place grad_value at the argmax position within a (kernel, kernel) zero array."
    block = np.zeros((kernel,kernel))
    block[argmax_index//kernel, argmax_index%kernel] = grad_value
    
    return block

# Step 24 - maxpool2d_backward
def maxpool2d_backward(d_out, cache):
    "scatter each d_out value to the cached argmax position in its window"
    
    
    N,C,H,W   = cache['x_shape']
    
    stride   = cache['stride']
    kernel = cache['kernel']
    argmax = cache['argmax']
    
    out_h = output_spatial_size(H, kernel, stride, 0)
    out_w = output_spatial_size(W, kernel, stride, 0)
    
    grad_res = np.zeros((N,C,H,W))
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    grad_res[n,c,i*stride:i*stride+kernel,j*stride:j*stride+kernel] += scatter_grad_window(d_out[n,c,i,j],argmax[n,c,i,j],kernel)
    return grad_res

# Step 25 - relu_forward
def relu_forward(x):
    # TODO: Compute the elementwise ReLU and cache the input for backprop.
    out = np.maximum(0,x)
    return out, {'x':x}

# Step 26 - relu_backward
def relu_backward(d_out, cache):
    # TODO: mask the upstream gradient by the positive entries of the cached input.
    x = cache['x']
   
    drelu = d_out*(x>0)
    return drelu

# Step 27 - flatten_forward
def flatten_forward(x):
    # TODO: reshape a 4D feature map into a 2D batch matrix and cache the original shape
    N,C,H,W = x.shape
    return x.reshape(N,-1),{'x_shape':x.shape}

# Step 28 - flatten_backward
import numpy as np

def flatten_backward(d_out, cache):
    # TODO: reshape the upstream gradient back to the original 4D feature map shape.
    N,C,W,H = cache['x_shape']
    return d_out.reshape(N,C,W,H)

# Step 29 - linear_forward (not yet solved)
# TODO: implement

# Step 30 - linear_grad_input (not yet solved)
# TODO: implement

# Step 31 - linear_grad_weights (not yet solved)
# TODO: implement

# Step 32 - linear_grad_bias (not yet solved)
# TODO: implement

# Step 33 - linear_backward (not yet solved)
# TODO: implement

# Step 34 - softmax_cross_entropy_forward (not yet solved)
# TODO: implement

# Step 35 - softmax_cross_entropy_backward (not yet solved)
# TODO: implement

# Step 36 - sgd_step (not yet solved)
# TODO: implement

# Step 37 - adam_update_m (not yet solved)
# TODO: implement

# Step 38 - adam_update_v (not yet solved)
# TODO: implement

# Step 39 - adam_bias_correct (not yet solved)
# TODO: implement

# Step 40 - adam_param_step (not yet solved)
# TODO: implement

# Step 41 - adam_step (not yet solved)
# TODO: implement

# Step 42 - init_conv_layer (not yet solved)
# TODO: implement

# Step 43 - init_linear_layer (not yet solved)
# TODO: implement

# Step 44 - init_lenet (not yet solved)
# TODO: implement

# Step 45 - forward_conv_block (not yet solved)
# TODO: implement

# Step 46 - forward_classifier_block (not yet solved)
# TODO: implement

# Step 47 - lenet_forward (not yet solved)
# TODO: implement

# Step 48 - backward_conv_block (not yet solved)
# TODO: implement

# Step 49 - backward_classifier_block (not yet solved)
# TODO: implement

# Step 50 - lenet_backward (not yet solved)
# TODO: implement

# Step 51 - lenet_predict (not yet solved)
# TODO: implement

# Step 52 - build_synthetic_image_dataset (not yet solved)
# TODO: implement

# Step 53 - shuffle_indices (not yet solved)
# TODO: implement

# Step 54 - train_test_split (not yet solved)
# TODO: implement

# Step 55 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 56 - train_step (not yet solved)
# TODO: implement

# Step 57 - train_one_epoch (not yet solved)
# TODO: implement

# Step 58 - train_loop (not yet solved)
# TODO: implement

# Step 59 - evaluate (not yet solved)
# TODO: implement

