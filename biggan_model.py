import tensorflow as tf
import tensorflow.keras.layers as layers
from inpainting_layers import InstanceNorm2D, upsampling2d_tpu, ConvSN2D
from utils import distributed, Reduction

# conditional batch norm -> instance norm
def biggan_residual_block(inputs, ch, downsampling, use_normalize):
    # main path 
    if use_normalize:
        x = InstanceNorm2D()(inputs)
    else:
        x = inputs
    x = layers.ReLU()(x)
    if downsampling < -1:
        x = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(x)
    x = ConvSN2D(ch, 3, padding="same")(x)
    if use_normalize:
        x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = ConvSN2D(ch, 3, padding="same")(x)
    if downsampling > 1:
        x = layers.AveragePooling2D(downsampling)(x)
    # residual path
    if downsampling < -1:
        r = layers.Lambda(upsampling2d_tpu, arguments={"scale": abs(downsampling)})(inputs)
    else:
        r = inputs
    r = ConvSN2D(ch, 1)(r)
    if downsampling > 1:
        r = layers.AveragePooling2D(downsampling)(r)
    return layers.Add()([x, r])

def image_to_image_generator():
    inputs = layers.Input((256, 256, 3))
    x = inputs
    encoders = []
    for ch in [256, 512, 1024]:
        x = ConvSN2D(ch, 3, padding="same", strides=2)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x)
        encoders.append(x)
    for d in [1, 1, 2, 2]:
        x = ConvSN2D(512, 3, padding="same", dilation_rate=d)(x)
        x = InstanceNorm2D()(x)
        x = layers.ReLU()(x) # (32, 32, 2048)
    x = layers.Concatenate()([x, encoders[-1]])
    x = biggan_residual_block(x, 1024, -2, True)  # (64, 64, 1024)
    x = layers.Concatenate()([x, encoders[-2]])
    x = biggan_residual_block(x, 512, -2, True)  # (128, 128, 512)
    x = layers.Concatenate()([x, encoders[-3]])
    x = biggan_residual_block(x, 256, -2, True)  # (256, 256, 256)
    x = InstanceNorm2D()(x)
    x = layers.ReLU()(x)
    x = ConvSN2D(3, 3, padding="same", activation="tanh")(x)
    model = tf.keras.models.Model(inputs, x)
    return model

def discriminator():
    inputs = layers.Input((256, 256, 3))
    x = biggan_residual_block(inputs, 256, 2, False) # (128, 128, 256)
    x = biggan_residual_block(x, 512, 2, False)  # (64, 64, 512)
    x = biggan_residual_block(x, 1024, 2, False)  # (32, 32, 1024)
    x = biggan_residual_block(x, 1024, 1, False)
    x = layers.ReLU()(x)
    x = ConvSN2D(1, 1)(x)
    model = tf.keras.models.Model(inputs, x)
    return model
