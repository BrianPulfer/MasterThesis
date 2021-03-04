import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to the h5 model to test")
    parser.add_argument("--tub320x240", type=str, help="Path to the test set / tub320x240")
    return dict(vars(parser.parse_args()))


def main():
    args = get_arguments()
    model_path, tub_path = args['model'], args['tub320x240']

    if not model_path or not tub_path:
        raise RuntimeError("Usage: --model <path/to/model> --tub320x240 <path/to/testtub>")

    if not os.path.isfile(model_path):
        raise RuntimeError(model_path + " is not a valid path")

    if not os.path.isdir(tub_path):
        raise RuntimeError(tub_path + " is not a valid path")

    # Loading model
    model = keras.models.load_model(model_path)
    model_input_size = model.layers[0]._batch_input_shape[1:3]

    # Looping through the images
    mse = 0
    for image_name in os.listdir(tub_path):
        if 'jpg' not in image_name.lower():
            continue

        # Defining paths
        image_path = os.path.join(tub_path, image_name)
        record_path = os.path.join(tub_path, "record_" + image_name.split("_")[0] + ".json")

        # Loading image and relative record
        image = Image.open(image_path).resize(model_input_size)
        record = json.load(open(record_path))

        # Getting the prediction and the error
        # prediction = model.predict(np.array([np.array(image)]))
        prediction = model(np.array([np.array(image).astype(np.float32)]))

        #print(prediction)
        #print(prediction[0])
        #print(prediction[0][0])
        print(prediction[0][0][0])

        # predicted_angle = ...
        # predicted_throttle = ...

        angle = record['user/angle']
        throttle = record['user/throttle']

        # print(angle)


if __name__ == '__main__':
    main()
