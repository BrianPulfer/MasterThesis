import os
import json
import argparse
import numpy as np

from thesis.xte_predictor.predict_real_xte import get_model, get_inputs, get_predictions

# Definitions
TUB = 'tub'
CROP = 'crop'
SIM = 'sim'
DEFINITIONS = [TUB, CROP, SIM]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--" + TUB, type=str, help="Path to the tub of the online test")
    parser.add_argument("--" + CROP, type=int, default=100, help="Number of top pixels to be cropped. Default: 100")
    parser.add_argument("--" + SIM, type=bool, default=False,
                        help="Whether the testing is in the simulator (use actual XTEs) or in the real-world (use XTE predictor).")
    args = vars(parser.parse_args())

    if args[TUB] is None:
        print("Usage: python online.py --tub <path> (--crop <crop value> --sim <boolean>)")
        exit()

    return args


def get_xtes(args):
    if args[SIM]:
        raise NotImplementedError("Online testing implemented only for real-world testing")
    xte_predictor = get_model()
    inputs = get_inputs(args[TUB], crop=args[CROP])
    return get_predictions(xte_predictor, inputs)


def get_tub_infos(tub_path):
    steers, throttles = [], []

    user = None

    for filename in os.listdir(tub_path):
        if '.json' not in filename.lower():
            continue

        if 'meta' in filename.lower():
            continue

        record = json.load(open(os.path.join(tub_path, filename)))

        if user is None:
            user = record['user/angle'] != 0.0 or record['user/throttle'] != 0.0

        steer = record['user/angle'] if user else record['pilot/angle']
        throttle = record['user/throttle'] if user else record['pilot/throttle']

        steers.append(steer)
        throttles.append(throttle)
    return np.array(steers), np.array(throttles)


def main():
    # Getting arguments
    args = get_args()

    # Getting XTE predicitons
    xtes = get_xtes(args)

    # Getting tub infos
    steers, throttles = get_tub_infos(args[TUB])

    # Printing online testing metrics
    print(
        "~XTE abs avg: {}\n"
        "~XTE abs max: {}\n"
        "Steering avg: {}\n"
        "Steering std: {}\n"
        "Throttle avg: {}\n"
        "Throttle std: {}\n".format(
            np.mean(np.abs(xtes)),
            np.max(np.abs(xtes)),
            np.mean(np.abs(steers)),
            np.std(np.abs(steers)),
            np.mean(np.abs(throttles)),
            np.std(np.abs(throttles))
        )
    )


if __name__ == '__main__':
    main()
