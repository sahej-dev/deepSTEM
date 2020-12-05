import tensorflow as tf
import numpy as np
import Network

def vector_out(i):
    v = np.zeros((10,1))
    v[i] = 1.0
    return v

(trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

imgX = 28
imgY = 28

trainX = [np.reshape(x, (imgX*imgY, 1)) for x in trainX]
trainX = [x / 255.0 for x in trainX]
trainY = [vector_out(x) for x in trainY]
train_data = (trainX, trainY)

testX = [np.reshape(x, (imgX*imgY, 1)) for x in testX]
testX = [x / 255.0 for x in testX]
# testY = [vector_out(x) for x in testY]
test_data = (testX, testY)

net = Network.Network([imgX*imgY, 30, 10])
net.sgd(train_data, epochs=30, batch_size=10, lr=3.0, test_dat=test_data)
# print(net.evaluate(test_data))