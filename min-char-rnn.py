import numpy as np
import sys

filename = 'input.txt'
    
with open(filename, 'r') as f:
    data = f.read()
    
# list of all unique characters in the dataset   
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

# mapping to get the unique index of a character and vice versa
char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

# Hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 16 # number of steps to unroll the RNN for
learning_rate = 1e-1

# Stop when processed this much data
MAX_DATA = 1000000

# Model parameters
Wxh = np.random.randn(hidden_size, vocab_size) * 0.01       # input to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01      # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size) * 0.01       # hidden to output
bh = np.zeros((hidden_size, 1))     # hidden bias
by = np.zeros((vocab_size, 1))      # output bias

def lossFun(inputs, targets, hprev):
    """Runs forward and backward passes through the RNN

    Args:
        inputs: List of integers. For some i, inputs[i] is the input
                character (encoded as an index into the ix_to_char map)
        targets: List of integers. For some i, targets[i] is the corrosponding
                 next character in the training data (similarly encoded)
        hprev: hidden_size x 1 array of previous hidden state
        returns: loss, gradients on model parameters and last hidden state
    """
    
    xs, hs, ys, ps = {}, {}, {}, {}
    # x is input vector
    # h = tanh(Whh * hprev + Wxh * x + bh)
    # y = Why * h + by
    # p = softmax(y)
    
    # Initial incoming state
    hs[-1] = np.copy(hprev)
    loss = 0
    # Forward pass
    for t in range(len(inputs)):
        # Input at time step t is xs[t].
        # Prepare a one-hot encoded vector of shape (vocab_size, 1)
        # inputs[t] is the index where the 1 goes.
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1
        
        # Compute h[t] from h[t-1] and x[t]
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        
        # Compute ps[t] - softmax probabilities for the output
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        # Calculate the cross-entropy loss
        loss += -np.log(ps[t][targets[t],0])
    
    # Backward pass: compute gradients going backwards.
    # Gradients are initialized to 0s and every time step contributes to them
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    
    # Initialize the incoming gradient of h to zero
    dhnext = np.zeros_like(hs[0])
    
    # The backward pass iterates over te input sequence backwards.
    for t in reversed(range(len(inputs))):
        # Backprop through the gradients of loss and softmax.
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        
        # Compute gradients for the Why and by parameters
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        
        # Backprop through the fully-connected layer (Why, by) to h. Also add up the
        # incoming gradient for h from the next cell.
        dh = np.dot(Why.T, dy) + dhnext
        
        # Backprop through tanh.
        dhraw = (1-hs[t] * hs[t]) * dh
        
        # Compute gradients for the dby, dWxh, Whh parameters
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        
        # Backprop the gradient to the incoming h, which will be used in the
        # previous time step.
        dhnext = np.dot(Whh.T, dhraw)
        
    # Gradient clipping to the range [-5, 5]
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out = dparam)
        
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
        
def sample(h, seed_ix, n):
    """Sample a sequence of integers from the model.
    
    Runs the RNN in forward mode for n steps; seed_ix is the seed letter for
    the first time step and h is the memory state.
    Returns a sequence of letters produced by the model
    """
    
    # Create a one-hot vector to represent the input
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    
    for t in range(n):
        # Run the forward pass only
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        
        # Sample fom the distribution produced by softmax
        ix = np.random.choice(range(vocab_size), p = p.ravel())
        
        # Prepare input for the next cell
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)
    return ixes 

# n is the iteration counter
# p is the input sequence pointer, at the beginning of each
# step it points at the sequence in the input that will be used for
# traingin this iteration
n, p = 0, 0

# Memory variables for Adagrad
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size) * seq_length

while p < MAX_DATA:
    # PRepare inputs 
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) # reset RNN memory
        p = 0 # go form start of data
        
    # in each step we unroll the RNN for seq_length cells, and present it 
    # with seq_length inputs and seq_length target outputs to learn
    inputs = [char_to_ix[ch] for ch in data[p: p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
    
    # Sample from the model now and then.
    if n% 1000 == 0:
        sample_ix = sample(hprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print(f'---\n {txt} \n---')
        
    # Forward seq_length characters throught the net and fetch the gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n% 200 == 0:
        print(f'iter {n} (p={p}, loss: {smooth_loss})')
        
    # Perform parameter update with Adagrad
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                    [dWxh, dWhh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += - learning_rate * dparam / np.sqrt(mem + 1e-8)
        
    p += seq_length
    n +=1
    
    
    
        