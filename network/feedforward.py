import numpy as np
import resource, sys
import random

def sig(x):
    return (1.0/(1.0 + np.exp(-x)))

def sig_prime(x):
    return sig(x)*(1-sig(x))
    #return np.exp(-x)/(1+np.exp(-x))**2

## função para construir matrix de alvos
def y(labels, epsilon, matrix_alvos = []):

    for i in labels:
        matrix_alvos.append([0+epsilon for j in range(10)])
        matrix_alvos[-1][i] = 1 - epsilon

    return matrix_alvos

class Network():

    def __init__(self, sizes):
        #incialização dos pesos, Xavier
        ## Primeiro valor médio e depois variancia 
        self.sizes = sizes
        self.weights = [np.random.normal(0, np.sqrt(1/(x)),(y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        #self.weights = [self.w1,self.w2]


    def SGD(self, x, y, batch, learn_rate):

        ##Shuffle 
        list_zipped = list(zip(x,y))
        np.random.shuffle(list_zipped)
        x,y = zip(*list_zipped)

        self.cost_a_epoch = 0
        self.accuracy = 0
        self.cc = 0

        ##Loop of examples
        for pos in range(0,len(x),batch):

                ##.T indica a transposta da matrix batch
                ## para que o dot product funcione normalmente.
                self.x = np.array(x[pos:pos+batch]).T

                ##Normalização das entradas
                self.x = self.x/255

                #Forward
                self.a = [self.x] # list to store all the activations, layer by layer
                self.z = [self.x] # list to store all the z vectors, layer by layer

                for w in self.weights:
                    self.z.append(np.dot(w, self.a[-1]))
                    self.a.append(sig(self.z[-1]))

                self.y = np.array(y[pos:pos+batch]).T

                ##Custo de cada exemplo separado por batch
                cost_x = sum((self.y - self.a[-1])**2)/2

                ##Custo de cada batch
                cost_a_batch = sum(cost_x)

                self.cost_a_epoch += cost_a_batch/60000
                #print(cost_a_batch)

                #self.weights = self.backprop(0.01)
                self.weights = self.target(learn_rate)

    def target(self, learn_rate):

        #Deltas Compute
        deltas = []
        deltas.append(-(self.y - self.a[-1])*sig_prime(self.z[-1]))

        taus = []
        for i,size in enumerate(self.sizes[:-1]):
            taus.append(np.maximum((1/np.sum(self.weights[i]**2,axis=0)).reshape(size,1),1))

        #Target Compute
        targets = []
        for i in range(1,len(self.sizes[:-1])):
            target = self.a[-i -1] - taus[-i]*np.dot(self.weights[-i].T,deltas[-1])
            target = np.minimum(target,1-0.01)
            target = np.maximum(target,-1+0.01)
            targets.append(target)
            deltas.append(-(targets[-1] - self.a[-i-1])*sig_prime(self.z[-i-1]))

        deltas.reverse()
        targets.reverse()
        w = []
        for i,d in enumerate(deltas):
            w.append(np.dot(d,self.a[i].T))

        ##Atualização dos pesos
        w_new = [w - learn_rate*w_prime for w,w_prime in zip(self.weights, w)]

        return w_new

    #Funcs for compute performance 
    def output(self, a):
        for w in self.weights:
            a = sig(np.dot(w,a))
        return a

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.output(np.array(x)/255)),y) for (x,y) in test_data]
        return sum([int(x == y) for (x,y) in test_results])








