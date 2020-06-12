import graph.api as gpi
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer


def init_net(layer0, hidden1, num_classes):
    l1_w = np.random.normal(size=(layer0, hidden1))
    l1 = gpi.variable(l1_w, name='l1')
    b1_w = np.random.normal(size=(hidden1))
    b1 = gpi.variable(b1_w, name='b1')
    l2_w = np.random.normal(size=(hidden1, num_classes))
    l2 = gpi.variable(l2_w, name='l2')
    b2_w = np.random.normal(size=(num_classes))
    b2 = gpi.variable(b2_w, name='b2')
    return [l1, b1, l2, b2]


def forward_pass(x, weights):
    assert len(weights) == 4
    assert x.shape[1] == weights[0].shape[0]
    a = gpi.dot(x, weights[0]) + weights[1]
    a = gpi.where(a > 0, a, 0)
    a = gpi.dot(a, weights[2]) + weights[3]
    return a


def main():
    print("Fetching MNIST...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print("Label Binarizing...")
    label_bin = LabelBinarizer()
    X = X / 255
    X_train, y = shuffle(X, y, random_state=0)
    y_train = label_bin.fit_transform(y) * 1.
    training_portion = np.floor(0.8 * len(X)).astype(np.int32)
    print("Train Set:", len(X_train))
    losses = []
    print("Initializing Weights...")
    weights = init_net(784, 64, 10)
    learning_rate = 1e-2
    iters = 555550
    print("   Learning Rate:", learning_rate)
    print("Total Iterations:", iters)
    print("Starting training...")
    for i in range(iters):
        random_sample = np.random.choice(len(X_train), 32)
        x, t = X_train[random_sample], y_train[random_sample]
        y = forward_pass(x, weights)
        loss = gpi.softmax_cross_entropy(y, t)
        losses.append(loss)
        print("\r[%4d/%4d]: Loss %.3f, Avg. Loss: %.3f" % (i+1, iters, loss,
                                                         np.mean(losses)))

        grads = loss.backward()
        weights[0] = weights[0] - learning_rate * grads['l1']
        weights[1] = weights[1] - learning_rate * grads['b1']
        weights[2] = weights[2] - learning_rate * grads['l2']
        weights[3] = weights[3] - learning_rate * grads['b2']


if __name__ == "__main__":
    main()
