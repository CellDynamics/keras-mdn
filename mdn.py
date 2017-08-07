from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import random_normal
import numpy as np
import math

def get_mixture_coef(output, numComonents=24, outputDim=1):
    out_pi = output[:,:numComonents]
    out_sigma = output[:,numComonents:2*numComonents]
    out_mu = output[:,2*numComonents:]
    out_mu = K.reshape(out_mu, [-1, numComonents, outputDim])
    out_mu = K.permute_dimensions(out_mu,[1,0,2])
    # use softmax to normalize pi into prob distribution
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu

def tf_normal(y, mu, sigma):
    oneDivSqrtTwoPI = 1 / math.sqrt(2*math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2,1,0])
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result)/2
    result = K.exp(result) * (1/(sigma + 1e-8))*oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])
    return result

def get_lossfunc(out_pi, out_sigma, out_mu, y):
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)

def mdn_loss(numComponents=24, outputDim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, numComponents, outputDim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss

class MixtureDensity(Layer):
    def __init__(self, kernelDim, numComponents, **kwargs):
        self.hiddenDim = 24
        self.kernelDim = kernelDim
        self.numComponents = numComponents
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, inputShape):
        self.inputDim = inputShape[1]
        self.outputDim = self.numComponents * (2 + self.kernelDim)
        self.Wh = self.add_weight(name='mdn_wh',
                                  shape=(self.inputDim, self.hiddenDim),
                                  initializer=random_normal(stddev=0.5))
        self.bh = self.add_weight(name='mdn_bh',
                                  shape=(self.hiddenDim),
                                  initializer=random_normal(stddev=0.5))
        self.Wo = self.add_weight(name='mdn_wo',
                                  shape=(self.hiddenDim, self.outputDim),
                                  initializer=random_normal(stddev=0.5))
        self.bo = self.add_weight(name='mdn_bo',
                                  shape=(self.outputDim),
                                  initializer=random_normal(stddev=0.5))

        super(Layer, self).build(inputShape)

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def compute_output_shape(self, inputShape):
        return (inputShape[0], self.outputDim)
