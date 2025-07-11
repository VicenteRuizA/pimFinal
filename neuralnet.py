from tensorflow.keras import layers, models, Input

def dcr_block(x, dilation_rate):
    shortcut = x
    x = layers.Conv3D(32, kernel_size=3, padding='same', dilation_rate=dilation_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Add()([shortcut, x])
    return x

def dscr_block_3d(x):
    shortcut = x
    x = layers.Conv3D(32, kernel_size=(3,1,1), padding='same', groups=32)(x)
    x = layers.Conv3D(32, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Add()([shortcut, x])
    return x

def build_parallel_riciannet(input_shape=(64, 64, 64, 1)):
    inputs = Input(shape=input_shape)

    dcr_path = layers.Conv3D(32, kernel_size=3, padding='same')(inputs)
    dscr_path = layers.Conv3D(32, kernel_size=3, padding='same')(inputs)

    for i in range(18):
        dcr_path = dcr_block(dcr_path, dilation_rate=(i % 3) + 1)
        dscr_path = dscr_block_3d(dscr_path)

    merged = layers.Concatenate()([dcr_path, dscr_path])
    x = layers.Conv3D(32, kernel_size=3, padding='same', activation='relu')(merged)
    residual = layers.Conv3D(1, kernel_size=3, padding='same')(x)

    output = layers.Subtract()([inputs, residual])
    return models.Model(inputs, output, name="3DParallelRicianNetLite")
