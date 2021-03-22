import os
import cv2
import argparse
import numpy as np
import pandas as pd

from keras import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.losses import MSE

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
            print("Usage: python train_sim2real_translator.py --data_dir <DIR>")
            exit()

    return args


def get_data(path, get_simulator_images=False):
    """Given the folder path, loads images from the /images/ sub-folder and labels (XTE) from the 'Frames' file."""

    # Path to images
    images_path = os.path.join(path, 'images/')

    # Loading all 'fake' images (translated from sim2real)
    images_names = list(os.listdir(images_path))
    images_names.sort(key=lambda filename: int(filename.split('_')[1]))
    images_array = np.array([cv2.imread(images_path + img_name) for img_name in images_names if 'fake' in img_name])

    # Loading all 'real' images (taken from simulator)
    simulator_images = None
    if get_simulator_images:
        simulator_images = [cv2.imread(images_path + img_name) for img_name in images_names if 'real' in img_name]
        simulator_images = np.array(simulator_images)

    # Loading all labels
    labels = pd.read_csv(os.path.join(path, "driving_log.csv"))['cte'].to_numpy()

    return images_array, labels, simulator_images


def get_model():
    # Input image is 256x256x3

    model = Sequential([
        Convolution2D(16, (5, 5)),  # 252x252x16
        Convolution2D(32, (5, 5)),  # 248x248x32
        MaxPooling2D((2, 2)),       # 124x124x32
        Convolution2D(32, (3, 3)),  # 122x122x32
        MaxPooling2D((2, 2)),       # 61x61x32
        Dropout(0.1),
        Convolution2D(64, (3, 3)),  # 59x59x64
        MaxPooling2D((2, 2)),       # 29x29x64
        Dropout(0.1),
        Convolution2D(64, (3, 3)),  # 27x27x64
        MaxPooling2D((2, 2)),       # 13x13x64
        Dropout(0.1),
        Flatten(),
        Dense(1024),
        Dense(128),
        Dense(1, activation='tanh')
    ])

    return model


def main():
    # Getting program arguments
    args = parse_arguments()

    # Collecting training set
    X, Y, _ = get_data(args[DATA_DIR])

    # Showing demo of the dataset
    # X, Y, SIM = get_data(args[DATA_DIR], True)
    # demo(X, Y, SIM)

    # Getting the model
    model = get_model()
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=MSE)
    model.fit(X, Y, batch_size=16, epochs=100)

    model.save("xte_predictor.h5")


def demo(sim_2_real_images, xtes, sim_images):
    for i in range(len(sim_2_real_images)):
        print("XTE: ", xtes[i])
        cv2.imshow("Sim Image", sim_images[i])
        cv2.imshow("Train Image", sim_2_real_images[i])
        cv2.waitKey(30)


if __name__ == '__main__':
    main()
