import numpy as np
from matplotlib import pyplot as plt
import random
from numpy.linalg import tensorinv
from numpy.random.mtrand import beta
import pandas as pd
import math


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """
    TODO: implement this function.

    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    dataset = pd.read_csv(filename)
    firstCol = dataset.columns[0]
    dataset = dataset.drop([firstCol], axis=1)
    dataset = np.array(dataset)
    return dataset


def print_stats(dataset, col):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on.
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    count = 0
    total = 0
    mean = 0
    for i in dataset.T[col]:
        count += 1
        total += i
    mean = total / count
    std_total = 0
    for j in dataset.T[col]:
        std_total += pow(j - mean, 2)
    std = math.sqrt(std_total / (count - 1))
    print(count)
    print('{0:.2f}'.format(mean))
    print('{0:.2f}'.format(std))


def regression(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    mse = 0
    mse_sub = 0
    for row in dataset:
        mse_sub += betas[0]
        for i in range(len(cols)):
            mse_sub += row[cols[i]] * betas[i + 1]
        mse_sub -= row[0]
        mse_sub = pow(mse_sub, 2)
        mse += mse_sub
        mse_sub = 0

    mse /= len(dataset)
    return mse


def gradient_descent(dataset, cols, betas):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    grads = []
    deri = 0
    ret = 0
    for betaInd in range(len(betas)):
        for row in dataset:
            deri += betas[0]
            for i in range(len(cols)):
                deri += row[cols[i]] * betas[i + 1]
            deri -= row[0]
            if betaInd != 0:
                deri *= row[cols[betaInd - 1]]
            ret += deri
            deri = 0
        ret *= 2
        ret /= len(dataset)
        grads.append(ret)
        ret = 0
    grads = np.array(grads)
    return grads


def iterate_gradient(dataset, cols, betas, T, eta):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """

    iteTime = 0
    for i in range(T):
        grads = gradient_descent(dataset, cols, betas)
        for index in range(len(betas)):
            betas[index] = betas[index] - (grads[index]*eta)
        iteTime = i + 1
        print(iteTime, '{0:.2f}'.format(
            regression(dataset, cols, betas)), end="")
        for beta in betas:
            print(" ", '{0:.2f}'.format(beta), end="")
        print("\n")


def compute_betas(dataset, cols):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    leng = len(dataset)
    dataset = dataset.T
    betas = []
    y = dataset[0][np.newaxis]
    y = y.T
    ones = np.ones([leng])
    xT = []
    xT.append(ones)
    for col in cols:
        xT.append(dataset[col])
    xT = np.array(xT)
    x = xT.T
    toInv = np.dot(xT, x)
    det = np.linalg.det(toInv)
    if det == 0:
        inv = np.linalg.inv(toInv)
    else:
        inv = np.linalg.pinv(toInv)

    temp = np.dot(inv, xT)
    betas = np.dot(temp, y)
    for beta in betas:
        beta = float(beta)
    dataset = dataset.T
    mse = regression(dataset, cols, betas)
    ret = []
    ret.append(mse[0])
    for beta in betas:
        ret.append(beta[0])
    ret = tuple(ret)
    return ret


def predict(dataset, cols, features):
    """
    TODO: implement this function.

    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    tuple = compute_betas(dataset, cols)

    result = tuple[1]
    for index in range(len(features)):
        result += tuple[index+2] * features[index]
    return result


def synthetic_datasets(betas, alphas, X, sigma):
    """
    TODO: implement this function.

    Input:
        betas  - parameters of the linear model
        alphas - parameters of the quadratic model
        X      - the input array (shape is guaranteed to be (n,1))
        sigma  - standard deviation of noise

    RETURNS:
        Two datasets of shape (n,2) - linear one first, followed by quadratic.
    """
    linear = np.zeros([len(X), 2])
    for i in range(len(X)):
        linear[i][0] = betas[0] + betas[1] * X[i] + np.random.normal(0, sigma)
        linear[i][1] = X[i]
    quad = np.zeros([len(X), 2])
    for i in range(len(X)):
        quad[i][0] = alphas[0] + alphas[1] * \
            pow(X[i], 2) + np.random.normal(0, sigma)
        quad[i][1] = X[i]
    return linear, quad


def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # TODO: Generate datasets and plot an MSE-sigma graph
    x = []
    for i in range(1000):
        x.append(random.randrange(-100, 100))

    betas = np.array([random.random(), random.random()])
    alphas = np.array([random.random(), random.random()])
    sigmas = []
    for i in range(-4, 6):
        sigmas.append(10**i)

    dic = {}
    for sigma in sigmas:
        dic[sigma] = synthetic_datasets(betas, alphas, x, sigma)

        # dic[sigma] = synthetic_datasets(np.array([0, 2]), np.array([0, 1]), x, sigma)

    arr1 = []
    arr2 = []
    for sigma in sigmas:
        arr1.append(compute_betas(dic[sigma][0], cols=[1])[0])
        arr2.append(compute_betas(dic[sigma][1], cols=[1])[0])
    plt.plot(sigmas, arr1, '-o')
    plt.plot(sigmas, arr2, '-o')

    plt.xlabel("Sigmas")
    plt.ylabel("MSE")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(["Linear", "Quadratic"])
    plt.savefig("mse.pdf")


if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
