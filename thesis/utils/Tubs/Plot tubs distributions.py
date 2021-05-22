
import os
import json
import numpy as np
import matplotlib.pyplot as plt



#tubs_path = input("Insert path to the folder containing the tubs: \t")
tubs_path = '/Users/brianpulfer/Desktop/USI/MAI-Thesis/Thesis/tests/data/real/'

for tub_name in os.listdir(tubs_path):
    tub_path = os.path.join(tubs_path, tub_name)

    if not os.path.isdir(tub_path):
        continue

    tub_throttles, tub_angles = [], []
    for record_name in os.listdir(tub_path):
        if 'record' not in record_name or 'json' not in record_name:
            continue

        record = json.load(open(os.path.join(tub_path, record_name)))
        throttle, angle = record['user/throttle'], record['user/angle']

        tub_throttles.append(throttle)
        tub_angles.append(angle)

    # Tub name to go in plot title
    tub_name = 'training' if 'train' in tub_name else 'test'

    """
    fig1, ax1 = plt.subplots()
    ax1.boxplot(tub_throttles)
    plt.show()

    fig1, ax1 = plt.subplots()
    ax1.boxplot(tub_angles)
    plt.show()

    """
    plt.hist(tub_throttles, 100, facecolor='red', alpha=0.75)
    plt.xlabel('Throttle')
    plt.ylabel('Cardinality')
    plt.title("Throttles on {} tub ".format(tub_name))
    plt.show()

    plt.hist(tub_angles, 100, facecolor='green', alpha=0.75)
    plt.xlabel('Angle')
    plt.ylabel('Cardinality')
    plt.title("Angles on {} tub ".format(tub_name))
    plt.show()

    """
    # Plotting throttles
    plt.plot(throttles_hist)


    # Plotting angles
    plt.plot(angles_hist)

    """
