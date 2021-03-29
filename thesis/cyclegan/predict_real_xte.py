# Imports
import os
import cv2
import pathlib
import argparse
import numpy as np
from tensorflow.python import keras

# Definitions
TUB_PATH = 'tub'
CROP = 'crop'
DEMO = 'demo'
STORE_IMAGES = 'store_images'


def get_program_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + TUB_PATH, type=str, help="Path to the 'real' tub from which XTEs have to be predicted")
    parser.add_argument("--" + CROP, type=int, default=100, help="Number of top pixels to be cropped. Default: 100")
    parser.add_argument("--" + DEMO, type=bool, default=True, help="Whether to play a demo of the predictions or not.")
    parser.add_argument("--" + STORE_IMAGES, type=bool, default=False, help="Whether to store demo images.")
    args = vars(parser.parse_args())

    if not args[TUB_PATH]:
        print("Usage: python predict_real_xte.py --tub <PATH>")
        exit()

    return args


def get_inputs(path, crop=100, size=(256, 256)):
    inputs = []

    images_names = [img_name for img_name in list(os.listdir(path)) if '.jpg' in img_name.lower()]

    try:
        images_names = sorted(images_names, key=lambda filename: int(filename.split("_")[0]))
    except ValueError:
        images_names = sorted(images_names, key=lambda filename: int(filename.split('_')[1].split('.jpg')[0]))

    print("\nCollecting {} images from {} ...".format(len(images_names), path))
    for img_name in images_names:
        # Reading the image
        img = cv2.imread(os.path.join(path, img_name))

        if img is not None:
            # Cropping top pixels
            img = img[crop:, :]

            # Resizing to be compatible with model's input size
            img = cv2.resize(img, size)

            # Appending
            inputs.append(img)
    print("{} images collected.".format(len(inputs)))
    return np.array(inputs)


def get_model():
    model_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'xte_predictor.h5')
    print("\nLoading model from {} ...".format(model_path))
    model = keras.models.load_model(model_path)
    print("Model loaded.")
    return model


def get_predictions(xte_predictor, inputs):
    print("\nGetting predictions from the model...")
    predictions = xte_predictor.predict(inputs)
    print("Predictions computed")
    return predictions


def show_demo(images, predicted_xtes, store, delay=100):
    print("\nDisplaying demo...")
    if store:
        counter = 0
        store_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'predictions/')
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
        print("Predictions will be stored under ", store_path)

    for img, xte in zip(images, predicted_xtes):
        if hasattr(xte, '__len__'):
            xte = xte[0]

        cv2.putText(img, "XTE: " + str(xte),
                    (64, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2)

        if store:
            cv2.imwrite(store_path + str(counter) + "_predicted_xte.jpg", img)
            counter += 1
        else:
            cv2.imshow("XTE Predictor", img)
            cv2.waitKey(delay)
    print("Demo displayed")


def store_predictions(predictions):
    file = open("predictions.txt")
    for p in predictions:
        file.write(str(p[0]) + "\n")
    file.close()


def main():
    # Getting the program arguments
    args = get_program_arguments()

    # Getting the model
    xte_predictor = get_model()

    # Getting inputs
    inputs = get_inputs(args[TUB_PATH], crop=args[CROP])

    # Running predictions
    predictions = get_predictions(xte_predictor, inputs)

    # Showing a demo
    if args[DEMO]:
        show_demo(inputs, predictions, args[STORE_IMAGES])

    # Storing predictions in txt file
    store_predictions(predictions)


if __name__ == '__main__':
    main()
