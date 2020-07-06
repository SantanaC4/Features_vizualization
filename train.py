from mnist import MNIST
import pickle
from network import feedforward as ff
from matplotlib import pyplot as plt

#Carregando dataset  
mndata = MNIST('./data')

images_training, labels_training = mndata.load_training()
images_testing, labels_testing = mndata.load_testing()

rede = ff.Network([784,128,10])

epoch = 1
learn_rate = 0.01
#####Treino por epoca########
for i in range(epoch):

    rede.SGD(images_training, ff.y(labels_training, 0.01), 100, learn_rate)
    print(rede.cost_a_epoch)

#########Teste###########
test_data = zip(images_testing, labels_testing)
desempenho = rede.evaluate(test_data)
print(desempenho)

# Save weights
with open("weights_{}ep_{}n_{}h_{:.2}%".format(epoch,rede.sizes[-2],len(rede.sizes[1:-1]),desempenho/100),"wb") as f:
        pickle.dump(rede.weights,f)


