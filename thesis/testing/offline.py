import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow import keras

# Definitions
TUB_PATH = 'tub'
CROP_TOP = 'crop'
MODEL_PATH = 'model'
DAVE2 = 'dave2'


def get_args():
    # Getting program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + TUB_PATH, type=str, help="Path to the tub")
    parser.add_argument("--" + CROP_TOP, type=int, default=100, help="Crop-top value. Default: 100")
    parser.add_argument("--" + MODEL_PATH, type=str, help="Path to the model to test")
    parser.add_argument("--" + DAVE2, type=bool, default=False, help="Whether the tested architecture is DAVE2")
    args = vars(parser.parse_args())

    if not args[TUB_PATH] or not args[MODEL_PATH]:
        print("Usage: python offline.py --{} <path> --{} <path>".format(TUB_PATH, MODEL_PATH))
        exit()

    return args


def load_model_and_check_sizes(model_path, tub_path, crop_top):
    # Loading model
    model = keras.models.load_model(model_path)

    # Getting model input size
    input_size = model.layers[0]._batch_input_shape[1:3]
    input_size = (input_size[1], input_size[0])

    # Getting dataset image size
    images_names = [name for name in sorted(os.listdir(tub_path)) if 'jpg' in name.lower()]
    first_image = Image.open(os.path.join(tub_path, images_names[0]))

    w, h = (first_image.size[0], first_image.size[1])
    print("Detected images_real size: ", first_image.size)
    print("Image size after top crop: ", (w, h - crop_top))
    print("Detected model input size: ", input_size)

    if input_size != (w, h - crop_top):
        print("WARNING: Input sizes do not match! Reshaping might yield bad predictions")
    else:
        print("Sizes seem to match\n")

    print(model.summary())
    return model, w, h


def get_dataset(tub_path, w, h, crop_top, dave2=False):
    # Getting file names
    images_names = [name for name in sorted(os.listdir(tub_path)) if 'jpg' in name.lower()]

    # Collecting dataset in a tensor
    n = len([i for i in os.listdir(tub_path) if 'jpg' in i.lower()])
    X, Y = [], []
    for image_name in tqdm(images_names):
        if 'jpg' not in image_name.lower():
            continue

        # Defining paths
        image_path = os.path.join(tub_path, image_name)
        record_path = os.path.join(tub_path, "record_" + image_name.split("_")[0] + ".json")

        # Loading image and relative record
        image = Image.open(image_path)
        resized = image.resize((w, h))
        normalized = np.array(resized) / 255.0
        cropped = normalized[crop_top:, ...]

        if dave2:
            cropped = cv2.cvtColor((cropped * 255).astype('uint8'), cv2.COLOR_RGB2YUV)
            cropped = cropped.astype('float') / 255

        record = json.load(open(record_path))

        # Loading labels
        angle = record['user/angle']
        throttle = record['user/throttle']

        # Adding data in the tensor
        X.append(np.array(cropped).astype(np.float32))
        Y.append(np.array([angle, throttle]))

    return np.array(X), np.array(Y)


def eval_model(angle_predictions, throttle_predictions, Y):
    mse_steer, mse_throttle = 0, 0
    mae_steer, mae_throttle = 0, 0
    for i in range(len(Y)):
        s, t = Y[i][0], Y[i][1]
        s_hat, t_hat = angle_predictions[i][0], throttle_predictions[i][0]

        mse_steer += (s_hat - s) ** 2
        mse_throttle += (t_hat - t) ** 2

        mae_steer += abs(s_hat - s)
        mae_throttle += abs(t_hat - t)

    # Getting average
    mse_steer /= len(Y)
    mse_throttle /= len(Y)
    mae_steer /= len(Y)
    mae_throttle /= len(Y)

    # Getting total loss
    loss = mse_steer + mse_throttle

    return loss, mse_steer, mse_throttle, mae_steer, mae_throttle


def main():
    # Getting arguments
    args = get_args()
    print(args)

    # Loading model
    model, w, h = load_model_and_check_sizes(args[MODEL_PATH], args[TUB_PATH], args[CROP_TOP])

    # Loading dataset
    X, Y = get_dataset(args[TUB_PATH], w, h, args[CROP_TOP], args[DAVE2])

    # Getting the predictions
    Y_hat = model.predict(X)
    angle_predictions = Y_hat[0]
    throttle_predictions = Y_hat[1]

    # Evaluating model offline
    loss, mse_steer, mse_throttle, mae_steer, mae_throttle = eval_model(angle_predictions, throttle_predictions, Y)

    # Printing
    print("Steering MSE: {}\t\tThrottle MSE: {}".format(mse_steer, mse_throttle))
    print("Steering MAE: {}\t\tThrottle MAE: {}".format(mae_steer, mae_throttle))
    print("Total LOSS: {}".format(loss))


if __name__ == '__main__':
    main()
