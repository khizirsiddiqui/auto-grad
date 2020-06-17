import graph.api as gpi
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from optim import SGDMomentum


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
    X, y = shuffle(X, y, random_state=0)
    y = label_bin.fit_transform(y) * 1.
    train_size = np.floor(0.8 * len(X)).astype(np.int32)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    print("Train Set:", len(X_train))
    losses = []
    print("Initializing Weights...")
    weights = init_net(784, 64, 10)
    iters = 150
    optimizer = SGDMomentum(weights)
    print("   Learning Rate:", optimizer.lr)
    print("Total Iterations:", iters)
    print("Starting training...")
    for i in range(iters):
        optimizer.zero_grad()
        random_sample = np.random.choice(len(X_train), 32)
        x, t = X_train[random_sample], y_train[random_sample]
        y = forward_pass(x, weights)
        loss = gpi.softmax_cross_entropy(y, t)
        losses.append(loss)
        print("\r[%3d/%3d]: Loss %.3f, Avg. Loss: %.3f" % (i+1, iters, loss,
                                                         np.mean(losses)))

        loss.backward()
        optimizer.step()

    correct = 0
    for x, t in zip(X_test, y_test):
        y_logits = forward_pass(x, weights)
        y = np.argmax(y_logits)
        correct += (y == np.argmax(t))
    print("Test Set Accuracy:", correct/len(X_test))

if __name__ == "__main__":
    main()
