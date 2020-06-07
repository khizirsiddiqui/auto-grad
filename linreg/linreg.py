from autograd import derivative, gradient

def mse(Y, y_pred):
    loss = 0
    for i in range(len(Y)):
        loss += (Y[i] - y_pred[i])**2
    return 0.002 * loss


def linear_regression(X, Y, steps=500, learning_rate=0.1, slope=0.1, intercept=0, verb=False):
    for i in range(steps):
        def loss_grad(slope, intercept):
            y_pred = [slope * x + intercept for x in X]
            return mse(Y, y_pred)

        y_pred = [x * slope + intercept for x in X]
        grad = gradient(loss_grad, [slope, intercept])
        slope -= learning_rate * grad[0]
        intercept -= learning_rate * grad[1]
        
        if verb:
            loss = mse(Y, y_pred)
            print("Step {}/{}: Loss {} ::: y_hat = {} X + {}"
                  .format(i+1, steps, loss, slope, intercept))

    return slope, intercept
