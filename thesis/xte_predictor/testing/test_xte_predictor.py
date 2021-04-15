import os
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras

MODEL = 'model'
SHOW_PREDICTIONS = 'show_predictions'


def get_arguments():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--' + MODEL, type=str, help='Path to the model')
    parser.add_argument('--' + SHOW_PREDICTIONS, type=bool, default=False, help='Path to the model')

    args = vars(parser.parse_args())

    if args[MODEL] is None:
        print("Usage: python test_xte_predictor.py --model <PATH>")
        exit()

    return args


def load_data(crop=100, size=(256, 256), shuffle=True):
    # Getting the labels (XTEs)
    frames = pd.read_csv('./xte_predictor_testset/xtes.csv', sep=';')
    labels = list(frames['xte'])
    filenames = frames['image_name'].to_list()

    # Getting the images
    images = []
    for path in [os.path.join('./xte_predictor_testset/images', filename) for filename in filenames]:
        image = cv2.imread(path)

        image = image[crop:, :]
        image = cv2.resize(image, size)
        images.append(image)

    return np.array(images), np.array(labels)


def show_predictions(X, Y, y_hat):
    for image, xte, xte_hat in zip(X, Y, y_hat):
        img = image
        cv2.putText(img, "~XTE: " + str(xte_hat),
                    (64, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)
        cv2.imshow("XTE: " + str(xte), img)
        cv2.waitKey(200)
    cv2.waitKey()


def plot_absolute_errors_barchart(errors):
    bin_size = 0.05
    bins = np.zeros(int(np.ceil(np.max(np.abs(errors)) / bin_size)))

    for error in np.sort(np.abs(errors)):
        idx = int(error / bin_size)
        bins[idx] += 1

    fig, ax = plt.subplots()
    x = np.arange(len(bins)) * bin_size
    rects1 = ax.bar(x, bins, 0.03, label='Absolute Error')

    ax.set_ylabel('Cardinality')
    ax.set_xlabel("|Error|")
    ax.set_title('XTE Predictor: Absolute error distribution on test set')
    ax.legend()

    fig.tight_layout()
    plt.show()


def print_mean_errors_for_classes(errors, Y):
    error_classes = [[], [], [], [], []]
    for err, Y in zip(errors, Y):
        label = int(Y)
        error_classes[label].append(np.abs(err))

    for i in [0, 1, 2, -1, -2]:
        err_class = error_classes[i]
        print("Mean absolute error for 'class' {} is: {:.4f} (class has {} images)".format(i, np.mean(err_class), len(err_class)))


def main():
    # Getting program args
    args = get_arguments()
    print(args)

    # Loading model
    model = keras.models.load_model(args[MODEL])

    # Loading the data
    X, Y = load_data()

    # Getting predictions
    y_hat = model.predict(X)

    if args[SHOW_PREDICTIONS]:
        # Showing predictions
        show_predictions(X, Y, y_hat)

    errors = []
    for i in range(len(y_hat)):
        error = y_hat[i] - Y[i]
        errors.append(error)
    errors = np.array(errors)

    print("MSE: ", np.mean(errors ** 2))
    print("MAE: ", np.mean(np.abs(errors)))
    print("Absolute Error mode: {}\n".format(np.percentile(np.abs(errors), 50)))

    # Plotting the barchart about the distribution of errors
    plot_absolute_errors_barchart(errors)

    # Mean errors for "classes" (-2, -1, 0, 1, 2)
    print_mean_errors_for_classes(errors, Y)


if __name__ == '__main__':
    main()
