from mnist import MNIST
from network import feedforward as ff
from matplotlib import pyplot as plt
from features import target_x
from os import listdir
from scipy  import spatial
import pickle
import sys
import seaborn as sns
import numpy as np

#Carregando dataset  
mndata = MNIST('../MNIST')

images_training, labels_training = mndata.load_training()
images_testing, labels_testing = mndata.load_testing()

#rede = ff.Network([784,512,10])
#epoch = 1
#learn_rate = 0.001
#########Treino por epoca########
#for i in range(epoch):
#
#    rede.SGD(images_training, ff.y(labels_training, 0.01), 100, learn_rate)
#    print(rede.cost_a_epoch)
#
#############Teste##########
#test_data = zip(images_testing, labels_testing)
#desempenho = rede.evaluate(test_data)
#print(desempenho)
#
## Save weights of training
#with open("../weights_backup/weights_{}ep_{}n_{}h_{}".format(epoch,rede.sizes[-2],len(rede.sizes[1:-1]),desempenho/100),"wb") as f:
#        pickle.dump(rede.weights,f)
#
#
########################features extraction##############################

#Save Weights list
list_w = listdir("../weights_backup")


x = np.zeros((784,1)).T

#Loop in weights list
for w_path in list_w:

    with open("../weights_backup/{}".format(w_path),"rb") as f:
        w = pickle.load(f)

    targets_x = []

    #Target neurons of output layer
    neurons = 10
    for i in range(neurons):
        y = target_x.one_hot(10,0.01)[i]
        targets_x.append(target_x.input(x,y,w))

    #Print input neurons expected 
    #fig, ax = plt.subplots(nrows=2, ncols=5, constrained_layout=True)
    #flag = 0
    #for i in range(2):
    #    for j in range(5):
    #        ax[i, j].imshow(targets_x[flag].reshape(28,28))
    #        ax[i, j].set_title('Neuron {}'.format(flag),fontsize=10)
    #        ax[i, j].axis('off')
    #        flag += 1

    #fig.suptitle('{}'.format(w_path), y=0.98 ,fontsize=16)
    #fig.savefig("../images/{}.png".format(w_path))
    #plt.close()

    #Cosine similarity
    cosine_similarity = []
    for i in range(10):
            cosine_similarity.append([1 - spatial.distance.cosine(targets_x[i],targets_x[j]) for j in range(10)])

    #Print right triangles of square matrix
    mask = np.zeros_like(cosine_similarity)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(cosine_similarity, mask=mask, vmax=.3, square=True,  cmap="YlGnBu")
        plt.title('{}'.format(w_path), y=0.98 ,fontsize=16)
        plt.savefig("../cosine_similarity/{}.png".format(w_path))
        plt.close()






















