import numpy as np
import random

class Network:
    def __init__(self, sizes):
        self.layers = len(sizes)
        self.sizes = sizes

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]


    def activation(self, z, type='sigmoid'):
        if type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            print('Error:', type, 'is an invalid activation function.')
            exit(1)


    def activation_prime(self, z, type='sigmoid'):
        if type == 'sigmoid':
            return self.activation(z) * (1 - self.activation(z))
        else:
            print('Error:', type, 'is an invalid activation function.')
            exit(1)


    def feed_forward(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(np.dot(w, a) + b)
        return a


    def sgd(self, train_dat, epochs, batch_size, lr, test_dat=None):
        for i in range(epochs):
            train_dat = shuffle(train_dat)
            batches = []
            for j in range(0, len(train_dat[0]), batch_size):
                batches.append((train_dat[0][j:j+batch_size], train_dat[1][j:j+batch_size]))

            for batch in batches:
                self.process_batch(batch, lr, batch_size)

            if test_dat:
                print('Epoch', i + 1, ':', self.evaluate(test_dat), '/', len(test_dat[0]))
            else:
                print('Epoch', i + 1, 'is complete')


    def process_batch(self, batch, lr, batch_size):
        dho_w = [np.zeros(w.shape) for w in self.weights]
        dho_b = [np.zeros(b.shape) for b in self.biases]

        i, j = batch
        for x, y in zip(i, j):
            delta_dho_w, delta_dho_b = self.backpropagation(x, y)
            # dho_w = dho_w + delta_dho_w
            # dho_b = dho_b + delta_dho_b
            dho_w = [dw + ddw for dw, ddw in zip(dho_w, delta_dho_w)]
            dho_b = [db + ddb for db, ddb in zip(dho_b, delta_dho_b)]

        self.weights = [w - (lr / batch_size) * dw for w, dw in zip(self.weights, dho_w)]
        self.biases  = [b - (lr / batch_size) * db for b, db in zip(self.biases, dho_b)]
        # self.weights = self.weights - (lr/batch_size) * dho_w
        # self.biases  = self.biases  - (lr/batch_size) * dho_b


    def backpropagation(self, x, y):
        a_vals = [x]
        z_vals = []

        dho_b = [np.zeros(b.shape) for b in self.biases]
        dho_w = [np.zeros(w.shape) for w in self.weights]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a_vals[-1]) + b
            z_vals.append(z)

            a = self.activation(z)
            a_vals.append(a)

        delta = self.cost_wrt_aL(
            a_vals[-1], y) * self.activation_prime(z_vals[-1])

        dho_b[-1] = delta
        dho_w[-1] = np.dot(delta, a_vals[-2].transpose())

        for l in range(2, self.layers):
            delta = np.dot(self.weights[-l+1].transpose(),
                           delta) * self.activation_prime(z_vals[-l])

            dho_b[-l] = delta
            dho_w[-l] = np.dot(delta, a_vals[-l-1].transpose())

        return (dho_w, dho_b)


    def cost_wrt_aL(self, y_prime, y):
        return y_prime - y


    def evaluate(self, test_dat):
        predictions = [(np.argmax(self.feed_forward(x)), y)
                        for x, y in zip(test_dat[0], test_dat[1])]

        return sum([y_prime == y for y_prime, y in predictions])


def shuffle(data):
    x, y = data
    z = range(len(x))
    random.shuffle(z)

    new_x = []
    new_y = []

    for i in z:
        new_x.append(x[i])
        new_y.append(y[i])

    return (new_x, new_y)