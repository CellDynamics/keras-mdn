from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math

def get_mixture_coef(output, num_components=24, output_dim=1):
    out_pi = output[:,:num_components]
    out_sigma = output[:,num_components:2*num_components]
    out_mu = output[:,2*num_components:]
    out_mu = K.reshape(out_mu, [-1, num_components, output_dim])
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

def mdn_loss(num_components=24, output_dim=1):
    def loss(y, output):
        out_pi, out_sigma, out_mu = get_mixture_coef(output, num_components, output_dim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss

class MixtureDensity(Layer):
    def __init__(self, kernel_dim, num_components, hidden_dim=24, init_stddev=0.075, **kwargs):
        self.kernel_dim = kernel_dim
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.init_stddev = init_stddev
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inputDim = input_shape[1]
        self.output_dim = self.num_components * (2 + self.kernel_dim)
        self.Wh = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.inputDim, self.hidden_dim)))
        self.bh = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.hidden_dim)))
        self.Wo = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.hidden_dim, self.output_dim)))
        self.bo = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.output_dim)))

        self.trainable_weights = [self.Wh,self.bh,self.Wo,self.bo]

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden,self.Wo) + self.bo
        return output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)
