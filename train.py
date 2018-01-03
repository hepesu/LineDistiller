import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Conv2D, Deconv2D, Activation, BatchNormalization, add
from keras.callbacks import ModelCheckpoint

SEED = 1

EPOCHS = 25
BATCH_SIZE = 4
LOAD_WEIGHTS = False

IMG_HEIGHT, IMG_WIDTH = 256, 256


def residual_block_downscaling(input_tensor, filters, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filter1, (1, 1), strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1))(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides)(input_tensor)

    x = add([x, shortcut])

    return x


def residual_block_upscaling(input_tensor, filters, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Deconv2D(filter1, (1, 1), strides=strides)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1))(x)

    shortcut = Deconv2D(filter3, (1, 1), strides=strides)(input_tensor)

    x = add([x, shortcut])

    return x


def residual_block(input_tensor, filters, shortcut_conv=False):
    filter1, filter2, filter3 = filters

    x = BatchNormalization()(input_tensor)
    x = Activation('relu')(x)
    x = Conv2D(filter1, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter2, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filter3, (1, 1))(x)

    if shortcut_conv:
        shortcut = Conv2D(filter3, (1, 1))(input_tensor)
        x = add([x, shortcut])
    else:
        x = add([x, input_tensor])

    return x


inputs = Input((None, None, 1))

x = residual_block(inputs, (8, 8, 32), True)
x = residual_block(x, (8, 8, 32))
x = residual_block(x, (8, 8, 32))
d1 = residual_block(x, (8, 8, 32))

x = residual_block_downscaling(d1, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
d2 = residual_block(x, (16, 16, 64))

x = residual_block_downscaling(d2, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
d3 = residual_block(x, (32, 32, 128))

x = residual_block_downscaling(d3, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))
x = residual_block(x, (64, 64, 256))

x = residual_block_upscaling(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
x = residual_block(x, (32, 32, 128))
u1 = residual_block(x, (32, 32, 128))
s1 = add([u1, d3])

x = residual_block_upscaling(s1, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
x = residual_block(x, (16, 16, 64))
u2 = residual_block(x, (16, 16, 64))
s2 = add([u2, d2])

x = residual_block_upscaling(s2, (8, 8, 32))
x = residual_block(x, (8, 8, 32))
x = residual_block(x, (8, 8, 32))
u3 = residual_block(x, (8, 8, 32))
s3 = add([u3, d1])

x = residual_block(s3, (4, 4, 16), True)
x = residual_block(x, (4, 4, 16))
x = residual_block(x, (4, 4, 16))

outputs = residual_block(x, (1, 1, 1), True)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

if LOAD_WEIGHTS:
    model.load_weights('model.h5')

model.compile(loss='MSE', optimizer='Adam')

datagen = image.ImageDataGenerator(
    rescale=1 / 255.,
    rotation_range=180,
    width_shift_range=0.5,
    height_shift_range=0.5,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect'
)

raw_generator = datagen.flow_from_directory(
    'data/raw',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

cs_generator = datagen.flow_from_directory(
    'data/contour_s',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

v_raw_generator = datagen.flow_from_directory(
    'data/v_raw',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

v_cs_generator = datagen.flow_from_directory(
    'data/v_contour_s',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='grayscale',
    seed=SEED,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=True
)

checkpointer = ModelCheckpoint(filepath='model.h5', verbose=1)

history = model.fit_generator(
    zip(raw_generator, cs_generator),
    steps_per_epoch=512 // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=zip(v_raw_generator, v_cs_generator),
    validation_steps=32 // BATCH_SIZE,
    callbacks=[checkpointer]
)

model.save('model_final.h5')
