from keras.layers import *
import keras.backend as K


class ExponentialMovingAverage(Layer):
    """Keeps an exponential average of the input variables.

    This is useful to mitigate overfitting
    (you could see it as a Bounded Rational Decision Maker where the running averages are the
    prior strategies.)

    As it is a regularization layer, it is only active at training time.

    # Arguments
        momentum: float, momentum of the epxonential decay
        initializers: string, how to initialize the variables

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.
    """

    def __init__(self, units=1, momentum=0.99, initiliazier='ones', **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self.supports_masking = True
        self.momentum = momentum
        self.units = units
        self.axis = -1
        self.initializer = initiliazier
        self.values = None
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[self.axis]

        self.values = self.add_weight(shape=(1, self.units),
                                      initializer=self.initializer,
                                      name='erm',
                                      trainable=False)

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]

        def update_erm():
            normed_training, mean, variance = K.normalize_batch_in_training(x=inputs, beta=None, gamma=None,
                                                                            reduction_axes=reduction_axes)
            self.add_update([K.moving_average_update(self.values,
                                                     mean,
                                                     self.momentum)],
                            inputs=inputs)
            return self.values

        return K.in_train_phase(update_erm(), self.values, training=training)

    def get_config(self):
        config = {'momentum': self.momentum}
        base_config = super(ExponentialMovingAverage, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))