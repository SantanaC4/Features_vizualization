import numpy as np

def sig(x):
    return (1.0/(1.0 + np.exp(-x)))

def sig_prime(x):
    return sig(x)*(1-sig(x))
    #return np.exp(-x)/(1+np.exp(-x))**2

def one_hot(neurons, epsilon):

    matrix_alvos = []
    for i in range(neurons):
        matrix_alvos.append(np.array([0+epsilon for j in range(neurons)]))
        matrix_alvos[-1][i] = 1 - epsilon

    return np.array([matrix_alvos]).T

def input(x,y,weights):

    sizes = [784]
    for w in weights:
        sizes.append(len(w))

    ##Normalização das entradas
    x = x.T

    #Forward
    a = [x] # list to store all the activations, layer by layer
    z = [x] # list to store all the z vectors, layer by layer

    for w in weights:
        z.append(np.dot(w, a[-1]))
        a.append(sig(z[-1]))

    #Deltas Compute
    deltas = []
    deltas.append(-(y - a[-1])*sig_prime(z[-1]))

    taus = []
    for i,size in enumerate(sizes[:-1]):
        taus.append(np.maximum((1/np.sum(weights[i]**2,axis=0)).reshape(size,1),1))

    #Target Compute
    targets = []
    for i in range(1,len(sizes)):
        target = a[-i -1] - taus[-i]*np.dot(weights[-i].T,deltas[-1])
        target = np.minimum(target,1-0.01)
        target = np.maximum(target,-1+0.01)
        targets.append(target)
        deltas.append(-(targets[-1] - a[-i-1])*sig_prime(z[-i-1]))

    targets.reverse()

    return targets[0]
