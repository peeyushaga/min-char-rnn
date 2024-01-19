import numpy as np
import sys

filename = "input.txt"

with open(filename, 'r') as f:
    data = f.read()
    
chars = list(set(data))
data_size = len(data)
V = vocab_size = len(chars)
print(f"data has {data_size} characters, {vocab_size} unique")

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}

H = hidden_size = 100
seq_length = 16
learning_rate = 0.1

HV = H + V

MAX_DATA = 1000000

Wf = np.random.randn(H, HV) * 0.01
bf = np.zeros((H,1))
Wi = np.random.randn(H, HV) * 0.01
bi = np.zeros((H, 1))
Wcc = np.random.randn(H, HV) * 0.01
bcc = np.zeros((H, 1))
Wo = np.random.randn(H, HV) * 0.01
bo = np.zeros((H, 1))
Wy = np.random.randn(V, H) * 0.01
by = np.zeros((V, 1))

def sigmoid(z):
    
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(z >= 0,
                        1/(1 + np.exp(-z)),
                        np.exp(z)/(1 + np.exp(z)))
        
def lossFun(inputs, targets, hprev, cprev):
    
    xs, xhs, ys, ps, hs, cs, fgs, igs, ccs, ogs = (
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
    
    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)
    
    loss = 0
    
    for t in range(len(inputs)):
        xs[t] = np.zeros((V, 1))
        xs[t][inputs[t]] = 1
        
        xhs[t] = np.vstack((xs[t], hs[t-1]))
        
        fgs[t] = sigmoid(np.dot(Wf, xhs[t]) + bf)
        igs[t] = sigmoid(np.dot(Wi, xhs[t]) + bi)
        ogs[t] = sigmoid(np.dot(Wo, xhs[t]) + bo)
        
        ccs[t] = np.tanh(np.dot(Wcc, xhs[t]) + bcc)
        
        cs[t] = fgs[t] * cs[t-1] + igs[t] * ccs[t]
        hs[t] = np.tanh(cs[t]) * ogs[t]
        
        ys[t] = np.dot(Wy, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        
        loss += -np.log(ps[t][targets[t], 0])

        
    dWf = np.zeros_like(Wf)
    dbf = np.zeros_like(bf)
    dWi = np.zeros_like(Wi)
    dbi = np.zeros_like(bi)
    dWcc = np.zeros_like(Wcc)
    dbcc = np.zeros_like(bcc)
    dWo = np.zeros_like(Wo)
    dbo = np.zeros_like(bo)
    dWy = np.zeros_like(Wy)
    dby = np.zeros_like(by)
    
    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])
    
    for t in reversed(range(len(inputs))):
        
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        
        dWy += np.dot(dy, hs[t].T)
        dby += dy
        
        dh = np.dot(Wy.T, dy) + dhnext
        
        dctanh = ogs[t] * dh
        
        dc = dctanh * (1-np.tanh(cs[t]) ** 2) + dcnext
        
        dhogs = dh * np.tanh(cs[t])
        dho = dhogs * ogs[t] * (1 - ogs[t])
        
        dWo += np.dot(dho, xhs[t].T) 
        dbo += dho
        
        dxh_from_o = np.dot(Wo.T, dho)
        
        dhf = cs[t-1] * dc * fgs[t] * (1 - fgs[t])
        dWf += np.dot(dhf, xhs[t].T)
        dbf += dhf
        dxh_from_f = np.dot(Wf.T, dhf)
        
        dhi = ccs[t] * dc * igs[t] * (1 - igs[t])
        dWi += np.dot(dhi, xhs[t].T)
        dbi += dhi
        dxh_from_i = np.dot(Wi.T, dhi)

        dhcc = igs[t] * dc * (1 - ccs[t] ** 2)
        dWcc += np.dot(dhcc, xhs[t].T)
        dbcc += dhcc
        dxh_from_cc = np.dot(Wcc.T, dhcc)     
        
        dxh = dxh_from_o + dxh_from_i + dxh_from_cc + dxh_from_f
        dhnext = dxh[V:, :]
        
        dcnext = fgs[t] * dc
        
    for dparam in [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    return (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
            hs[len(inputs)-1], cs[len(inputs)-1])
    
def sample(h, c, seed_ix, n):
    
    x = np.zeros((V, 1))
    x[seed_ix] = 1
    ixes = []
    
    for t in range(n):
        
        xh = np.vstack((x, h))
        fg = sigmoid(np.dot(Wf, xh) + bf)
        ig = sigmoid(np.dot(Wi, xh) + bi)
        og = sigmoid(np.dot(Wo, xh) + bo)
        cc = np.tanh(np.dot(Wcc, xh) + bcc)
        c = fg * c + ig * cc
        h = np.tanh(c) * og
        y = np.dot(Wy, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        
        ix = np.random.choice(range(V), p=p.ravel())
        x = np.zeros((V, 1))
        x[ix] = 1
        ixes.append(ix)
        
    return ixes

n, p = 0, 0

mWf = np.zeros_like(Wf)
mbf = np.zeros_like(bf)
mWi = np.zeros_like(Wi)
mbi = np.zeros_like(bi)
mWcc = np.zeros_like(Wcc)
mbcc = np.zeros_like(bcc)
mWo = np.zeros_like(Wo)
mbo = np.zeros_like(bo)
mWy = np.zeros_like(Wy)
mby = np.zeros_like(by)
smooth_loss = -np.log(1.0/V) * seq_length

while p < MAX_DATA:
    
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((H, 1))
        cprev = np.zeros((H, 1))
        p = 0
        
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    
    if n % 1000 == 0:
        sample_ix = sample(hprev, cprev, inputs[0], 200)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print("---------------------------------")
        print(txt)
        print("---------------------------------")
        
    (loss, dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby,
     hprev, cprev) = lossFun(inputs, targets, hprev, cprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n% 200 == 0:
        print(f"iter {n} (p = {p}), loss {smooth_loss}")
      
    for param, dparam, mem in zip(
            [Wf, bf, Wi, bi, Wcc, bcc, Wo, bo, Wy, by],
            [dWf, dbf, dWi, dbi, dWcc, dbcc, dWo, dbo, dWy, dby],
            [mWf, mbf, mWi, mbi, mWcc, mbcc, mWo, mbo, mWy, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
        
    p+=seq_length
    n+=1
    
    