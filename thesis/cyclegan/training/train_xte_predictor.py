import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from thesis.own_models import get_default_donkeycar_model

# DEFINITIONS
DATA_DIR = 'data_dir'

PROGRAM_ARGUMENTS = [DATA_DIR]


def parse_arguments():
    """Parses the program arguments and returns the dictionary to be referred to with the Definitions"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--" + DATA_DIR, type=str, help="Path to the directory containing the dataset")

    args = vars(parser.parse_args())

    for prog_arg in PROGRAM_ARGUMENTS:
        if not args[prog_arg]:
            print("Usage: python train_xte_predictor.py --data_dir <DIR>")
            exit()

    return args


def get_data(path, shuffle=True):
    """Given the folder path, loads images from the /images/ sub-folder and labels (XTE) from the 'Frames' file."""

    # Loading all labels
    frames = pd.read_csv(os.path.join(path, "driving_log.csv"), sep=',')
    labels = frames['cte'].to_numpy()
    filenames = frames['center'].to_list()

    images = []
    for filename in filenames:
        filename = filename.split("/")[-1].split(".jpg")[0]
        image_path = os.path.join(path, 'images/', filename+"_fake.png")
        images.append(cv2.imread(image_path))

    if shuffle:
        combined = list(zip(images, labels))
        random.shuffle(combined)
        images, labels = zip(*combined)

    return np.array(images), labels


def get_model():
    from donkeycar.parts.keras import Input, Convolution2D, Dropout, Dense, Flatten, Model
    drop = 0.1

    img_in = Input(shape=(256, 256, 3), name='img_in')
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
        Dense(1, activation='linear', name='n_outputs0')(x)
    ]

    # for i in range(2):
    #   outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    return Model(inputs=[img_in], outputs=outputs)


def main():
    # Getting program arguments
    args = parse_arguments()

    # Collecting training set
    X, Y = get_data(args[DATA_DIR])

    percentage = 0.95
    split = int(percentage * len(X))
    x_train, y_train = X[:split], Y[:split]
    x_val, y_val = X[split:], Y[split:]

    # Getting the model
    model = get_model()
    model.compile(optimizer=Adam(lr=0.00001), loss='mse')
    history = model.fit(x_train,
              y_train,
              batch_size=16,
              epochs=100,
              validation_data=(x_val, y_val),
              callbacks=[
                  EarlyStopping('val_loss', patience=10),
                  ModelCheckpoint('../xte_predictor_old.h5', 'val_loss', save_weights_only=False)
              ]
              ).history

    plt.plot(np.arange(len(history['loss'])), history['loss'], 'r-', label="Train Loss")
    plt.plot(np.arange(len(history['val_loss'])), history['val_loss'], 'g-', label="Validation Loss")
    plt.show()

    model.save("xte_predictor_old.h5")


if __name__ == '__main__':
    main()
