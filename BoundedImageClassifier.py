from keras.models import Model
from keras.optimizers import *
from ExponentialMovingAverageLayer import *
import tensorflow as tf
from keras import backend as K


def create_image_classifier(input_shape, num_classes, beta):
    def cat_cross_beta(beta):
        beta_var = K.variable(beta)

        def categorical_crossentropy(target, output, from_logits=False, b=beta_var):
            """Categorical crossentropy between an output tensor and a target tensor.

            # Arguments
                target: A tensor of the same shape as `output`.
                output: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `output` is expected to be the logits).
                from_logits: Boolean, whether `output` is the
                    result of a softmax, or is a tensor of logits.

            # Returns
                Output tensor.
            """
            # Note: tf.nn.softmax_cross_entropy_with_logits
            # expects logits, Keras expects probabilities.
            clipped_y = K.clip(output, K.epsilon(), 1)
            kl_loss = clipped_y * (K.log(clipped_y / K.clip(erm, K.epsilon(), 1)))
            kl_loss = (K.variable(1.0) / K.variable(beta)) * K.mean(kl_loss)

            if not from_logits:
                # scale preds so that the class probas of each sample sum to 1
                output /= tf.reduce_sum(output,
                                        len(output.get_shape()) - 1,
                                        True)
                # manual computation of crossentropy
                output = tf.clip_by_value(output, K.epsilon(), 1. - K.epsilon())
                return - tf.reduce_sum(target * tf.log(output),
                                       len(output.get_shape()) - 1) - kl_loss
            else:
                return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                               logits=output) - kl_loss

        return categorical_crossentropy

    inlayer = Input(shape=input_shape)
    h = Conv2D(32, kernel_size=(3, 3),
               activation='relu', kernel_initializer='he_normal', use_bias=True)(inlayer)
    h = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', use_bias=True)(h)
    h = Flatten()(h)
    h = Dense(128, activation='tanh', kernel_initializer='he_normal', use_bias=True)(h)
    h = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', use_bias=True)(h)
    erm = ExponentialMovingAverage(units=num_classes, name="conv_erm")(h)
    model = Model(inlayer, [h, erm])

    model.compile(loss=[cat_cross_beta(beta=beta), None],
                  optimizer=Adam(),
                  metrics=['acc'])
    return model
