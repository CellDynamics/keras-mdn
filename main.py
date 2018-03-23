from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.backend import eval
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # <-- Note the capitalization!
from mdn import *


def generate(output, testSize, numComponents=24, outputDim=1, M=1):  # XXX What is M
    out_pi, out_sigma, out_mu = get_mixture_coef(output, num_components=numComponents, output_dim=outputDim)
    # convert tensors to numpy
    out_pi = eval(out_pi)
    out_sigma = eval(out_sigma)
    out_mu = eval(out_mu)
    result = np.random.rand(testSize, M, outputDim)
    rn = np.random.randn(testSize, M)
    mu = 0
    std = 0
    idx = 0
    for j in range(0, M):
        for i in range(0, testSize):
            for d in range(0, outputDim):
                idx = np.random.choice(numComponents, 1, p=out_pi[i])
                mu = out_mu[idx, i, d]
                std = out_sigma[i, idx]
                result[i, j, d] = mu + rn[i, j] * std
    return result


def oneDim2OneDim():
    sampleSize = 2500
    numComponents = 24
    outputDim = 1
    x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sampleSize))).T
    r_data = np.float32(np.random.normal(size=(sampleSize, 1)))
    y_data = np.float32(np.sin(0.75 * x_data) * 7.0 + x_data * 0.5 + r_data * 1.0)
    # invert training data
    temp_data = x_data
    x_data = y_data
    y_data = temp_data

    model = Sequential()
    model.add(Dense(256, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(MixtureDensity(outputDim, numComponents, init_stddev=0.5))

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(num_components=numComponents), optimizer=opt)
    model.fit(x_data, y_data, batch_size=x_data.size, nb_epoch=10000, verbose=0)

    x_test = np.float32(np.arange(-15.0, 15.0, 0.05))
    x_test = x_test.reshape(x_test.size, 1)
    y_test = generate(model.predict(x_test), x_test.size, numComponents=numComponents)

    plt.figure(figsize=(8, 8))
    plt.plot(x_data, y_data, 'ro', x_test, y_test[:, :, 0], 'bo', alpha=0.3)
    plt.show()


def oneDim2TwoDim():
    sampleSize = 250
    numComponents = 24
    outputDim = 2

    z_data = np.float32(np.random.uniform(-10.5, 10.5, (1, sampleSize))).T
    r_data = np.float32(np.random.normal(size=(sampleSize, 1)))
    x1_data = np.float32(np.sin(0.75 * z_data) * 7.0 + z_data * 0.5 + r_data * 1.0)
    x2_data = np.float32(np.sin(0.5 * z_data) * 7.0 + z_data * 0.5 + r_data * 1.0)
    x_data = np.dstack((x1_data, x2_data))[:, 0, :]

    model = Sequential()
    model.add(Dense(128, input_shape=(1,)))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(MixtureDensity(outputDim, numComponents))

    opt = Adam(lr=0.001)
    model.compile(loss=mdn_loss(numComponents=24, outputDim=outputDim), optimizer=opt)
    model.fit(z_data, x_data, batch_size=x_data.size, nb_epoch=10000, verbose=1)

    x_test = np.float32(np.arange(-15.0, 15.0, 0.1))
    x_test = x_test.reshape(x_test.size, 1)
    y_test = generate(model.predict(x_test),
                      x_test.size,
                      numComponents=numComponents,
                      outputDim=outputDim)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(y_test[:, 0, 0], y_test[:, 0, 1], x_test, c='r')
    ax.scatter(x1_data, x2_data, z_data, c='b')
    ax.legend()
    plt.show()


# oneDim2OneDim()
# oneDim2TwoDim()
