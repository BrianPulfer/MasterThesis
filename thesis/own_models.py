from donkeycar.parts.keras import KerasPilot, Model
from donkeycar.parts.keras import Convolution2D, Lambda, Flatten, Dense, Dropout, Input, MaxPooling2D

# Definitions
DAVE2 = 'dave2'
CHAUFFEUR = 'chauffeur'
EPOCH = 'epoch'
RAMBO = 'rambo'
DEFAULT_DONKEY = 'defaultdonkey'


def get_own_model(model_name):
    model = None
    if model_name == DAVE2:
        model = get_dave2_model()
    elif model_name == CHAUFFEUR:
        model = get_chaffeur_model()
    elif model_name == EPOCH:
        model = get_epoch_model()
    elif model_name == RAMBO:
        model = get_rambo_model()
    elif model_name == DEFAULT_DONKEY:
        model = get_default_donkeycar_model()
    else:
        return None

    class OwnModel(KerasPilot):
        def __init__(self, *args, **kwargs):
            super(OwnModel, self).__init__(*args, **kwargs)
            self.model = model
            self.compile()

        def compile(self):
            self.model.compile(optimizer=self.optimizer, loss='mse')

        def run(self, img_arr):
            img_arr = img_arr.reshape((1,) + img_arr.shape)
            outputs = self.model.predict(img_arr)
            steering = outputs[0]
            throttle = outputs[1]
            return steering[0][0], throttle[0][0]

    return OwnModel


def get_dave2_model(input_shape=(66, 200, 3)):
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    # Normalization is carried out in donkey car train data generator
    # x = Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape, name='lambda_norm')(x)
    x = Lambda(lambda x: x * 255 / 127.5 - 1., input_shape=input_shape, name='lambda_norm')(x)

    # 5x5 Convolutional layers with stride of 2x2
    x = Convolution2D(24, (5, 5), strides=(2, 2), name='conv1', activation='elu')(x)
    x = Convolution2D(36, (5, 5), strides=(2, 2), name='conv2', activation='elu')(x)
    x = Convolution2D(48, (5, 5), strides=(2, 2), name='conv3', activation='elu')(x)

    # 3x3 Convolutional layers with stride of 1x1
    x = Convolution2D(64, (3, 3), strides=(1, 1), name='conv4', activation='elu')(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), name='conv5', activation='elu')(x)

    # Dropout ?
    # x = Dropout(0.05)(x)

    # Flatten before passing to Fully Connected layers
    x = Flatten()(x)

    # Three fully connected layers
    x = Dense(100, name='fc1', activation='elu')(x)
    x = Dropout(.5, name='do1')(x)
    x = Dense(50, name='fc2', activation='elu')(x)
    x = Dropout(.5, name='do2')(x)
    x = Dense(10, name='fc3', activation='elu')(x)
    x = Dropout(.5, name='do3')(x)

    # Output layer with tanh activation
    outputs = [
        Dense(1, activation='tanh', name='n_outputs0')(x),
        Dense(1, activation='sigmoid', name='n_outputs1')(x),
    ]

    #for i in range(2):
    #    outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    return Model(inputs=[img_in], outputs=outputs)


def get_chaffeur_model(input_shape=(120, 320, 3)):
    from donkeycar.parts.keras import Dropout as SpatialDropout2D

    def get_convolution_kernels(n, kernel_size):
        return Convolution2D(n,
                             kernel_size,
                             kernel_initializer="he_normal",
                             bias_initializer="he_normal",
                             activation='relu',
                             padding='same')

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    # (Convolution -> Spatial Dropout -> Max Pooling) x5
    x = Convolution2D(16, (5, 5), input_shape=input_shape, kernel_initializer="he_normal", bias_initializer="he_normal",
                      activation='relu', padding='same')(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(20, (5, 5))(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(40, (3, 3))(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(60, (3, 3))(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = get_convolution_kernels(80, (2, 2))(x)
    x = SpatialDropout2D(0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Flattening and dropping-out
    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # Classification
    outputs = [
        Dense(1, activation='tanh', name='n_outputs0')(x),
        Dense(1, activation='sigmoid', name='n_outputs1')(x),
    ]

    #for i in range(2):
    #    outputs.append(Dense(1, name='n_outputs' + str(i), kernel_initializer='he_normal')(x))

    return Model(inputs=[img_in], outputs=outputs)


def get_epoch_model(input_shape=(66, 200, 3)):
    img_in = Input(shape=input_shape)
    x = img_in

    # Normalization
    # x = Lambda(lambda x: x / 127.5 - 1.0)(x)
    # x = Lambda(lambda x: x * 255 / 127.5 - 1.0)(x)

    x = Convolution2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Convolution2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(.5)(x)

    outputs = [
        Dense(1, activation='tanh', name='n_outputs0')(x),
        Dense(1, activation='sigmoid', name='n_outputs1')(x),
    ]

    # for i in range(2):
    #    outputs.append(Dense(1, name='n_outputs' + str(i))(x))

    return Model(inputs=[img_in], outputs=outputs)


def get_rambo_model():
    raise NotImplementedError("Rambo model is not yet implemented")


def get_default_donkeycar_model(input_shape=(140, 320, 3)):
    drop = 0.1

    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    x = Convolution2D(24, (5, 5), strides=(2, 2), activation='relu', name="conv2d_1")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(32, (5, 5), strides=(2, 2), activation='relu', name="conv2d_2")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (5, 5), strides=(2, 2), activation='relu', name="conv2d_3")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_4")(x)
    x = Dropout(drop)(x)
    x = Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', name="conv2d_5")(x)
    x = Dropout(drop)(x)

    x = Flatten(name='flattened')(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(drop)(x)

    outputs = [
        Dense(1, activation='tanh', name='n_outputs0')(x),
        Dense(1, activation='sigmoid', name='n_outputs1')(x),
    ]

    # for i in range(2):
    #   outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    return Model(inputs=[img_in], outputs=outputs)
