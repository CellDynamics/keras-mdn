from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import math
import tensorflow as tf


def get_mixture_coef(output, num_components=24, output_dim=1):
    """Extract MDN model coefficients.

    Split output from previous layer into three components of Mixture Model.
    """
    out_pi = output[:, :num_components]  # mixing coefficient
    out_sigma = output[:, num_components:2 * num_components]
    out_mu = output[:, 2 * num_components:]
    out_mu = K.reshape(out_mu, [-1, num_components, output_dim])
    out_mu = K.permute_dimensions(out_mu, [1, 0, 2])
    # use softmax to normalize pi into prob distribution (from paper)
    max_pi = K.max(out_pi, axis=1, keepdims=True)
    out_pi = out_pi - max_pi  # XXX Why max should be 1 before computing softmax?
    out_pi = K.exp(out_pi)
    normalize_pi = 1 / K.sum(out_pi, axis=1, keepdims=True)
    out_pi = normalize_pi * out_pi
    # use exponential to make sure sigma is positive (from paper)
    out_sigma = K.exp(out_sigma)
    return out_pi, out_sigma, out_mu


def tf_normal(y, mu, sigma):
    """Compute Eq. 23 from Bishop paper."""
    oneDivSqrtTwoPI = 1 / math.sqrt(2 * math.pi)
    result = y - mu
    result = K.permute_dimensions(result, [2, 1, 0])  # XXX Why this? Not in tensoflow implementaiton
    result = result * (1 / (sigma + 1e-8))
    result = -K.square(result) / 2
    result = K.exp(result) * (1 / (sigma + 1e-8)) * oneDivSqrtTwoPI
    result = K.prod(result, axis=[0])  # XXX Why this? Not in tensoflow implementaiton
    return result


def get_lossfunc(out_pi, out_sigma, out_mu, y):
    """Compute Eq. 29 from Bishop paper."""
    # out_pi = tf.Print(out_pi, [out_pi], message='out_pi', summarize=1000)
    # out_sigma = tf.Print(out_sigma, [out_sigma], message='out_sigma', summarize=1000)
    # out_mu = tf.Print(out_mu, [out_mu], message='out_mu', summarize=1000)
    result = tf_normal(y, out_mu, out_sigma)
    result = result * out_pi
    result = K.sum(result, axis=1, keepdims=True)
    result = -K.log(result + 1e-8)
    return K.mean(result)  # XXX Why mean here? Is it along whole batch? It is in TF implementation


def mdn_loss(num_components=24, output_dim=1):
    """Provide MDN loss function for training."""
    def loss(y, output):
        """Compute model coefficient based on MDN outputed from network and real network target.

        Create mapping between MDN model and real network targets.
        """
        out_pi, out_sigma, out_mu = get_mixture_coef(output, num_components, output_dim)
        return get_lossfunc(out_pi, out_sigma, out_mu, y)
    return loss


class MixtureDensity(Layer):
    """Mixture Density Layer."""

    def __init__(self, kernel_dim, num_components, hidden_dim=24, init_stddev=0.075, **kwargs):
        """Construct the layer.

        MDN layer is built on the top of simple multi-layer perceptron model with one hidden layer and linear outputs.
        see page 7 in the paper. This works like normal neuron layer, different is only interpretation of outputs, which
        are not estimated target values but coefficients of MDN model. They must be 'decoded' by:
        - 'get_mixture_coef' to get MDN model coefficients
        - 'mdn_loss' - to get error between expected target and network output. Loss function consumes MDN coefficients.

        Args:
            kernel_dim (int)        - number of outputs from layer (network) (see page 7)
            num_components (int)    - number of mixture components (m in Eq. 22)
            hidden_dim (int)        - dimension of hidden layer in our model.
            init_stddev (float)     - standard deviation of normal distribution used for initialising trainable
                                      parameters
        """
        self.kernel_dim = kernel_dim
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        self.init_stddev = init_stddev
        super(MixtureDensity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inputDim = input_shape[1]
        self.output_dim = self.num_components * (2 + self.kernel_dim)  # XXX Page 7, green comment not sure why?
        # parameters of hidden layer - trainable
        self.Wh = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.inputDim, self.hidden_dim)))
        self.bh = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.hidden_dim)))
        # parameters of linear output layer
        self.Wo = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.hidden_dim, self.output_dim)))
        self.bo = K.variable(np.random.normal(scale=self.init_stddev,
                                              size=(self.output_dim)))

        self.trainable_weights = [self.Wh, self.bh, self.Wo, self.bo]
        self.built = True

    def call(self, x, mask=None):
        hidden = K.tanh(K.dot(x, self.Wh) + self.bh)
        output = K.dot(hidden, self.Wo) + self.bo
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {
            'kernel_dim': self.kernel_dim,
            'num_components': self.num_components,
            'hidden_dim': self.hidden_dim,
            'init_stddev': self.hidden_dim,
        }
        base_config = super(MixtureDensity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
