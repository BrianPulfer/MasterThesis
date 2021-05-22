import os
import cv2
import pathlib
import argparse
import numpy as np
from skimage.metrics import structural_similarity as ssim


def get_ssim_scores(tub1, tub2):
    scores = []

    tub1_img_names = list(os.listdir(tub1))
    tub1_img_names = [img_name for img_name in tub1_img_names if '.jpg' or '.png' in img_name]
    tub1_img_names = sorted(tub1_img_names, key=lambda name: int(name.split("_")[0]))

    tub2_img_names = list(os.listdir(tub2))
    tub2_img_names = [img_name for img_name in tub2_img_names if '.jpg' or '.png' in img_name]
    tub2_img_names = sorted(tub2_img_names, key=lambda name: int(name.split("_")[0]))

    for img1_name, img2_name in zip(tub1_img_names, tub2_img_names):
        # Defining image paths
        img1_path = os.path.join(tub1, img1_name)
        img2_path = os.path.join(tub2, img2_name)

        # Loading images_real
        img_1 = cv2.imread(img1_path)
        img_2 = cv2.imread(img2_path)

        # Preprocessing non-artificial images_real
        img_1 = img_1[100:, :]
        img_1 = cv2.resize(img_1, (256, 256))

        # Checking dimensions
        if img_1.shape != img_2.shape:
            print(f"Image size mismatch: comparing images of size {img_1.shape} and {img_2.shape}")
            exit()

        # Getting the Structural Similarity Index (SSI)
        scores.append(ssim(img_1, img_2, multichannel=True))
    return scores


def print_stats(scores):
    print(f"Average Structural Similarity Index Measure: {np.mean(scores)}")
    print(f"Std of Structural Similarity Index Measures: {np.std(scores)}")
    print(f"Minimum similarity: {np.min(scores)}")
    print(f"Maximum similarity: {np.max(scores)}")
    print(f"Mode similarity: {np.percentile(scores, 0.5)}")


def main():
    # Defining path to real and pseudo-real images_real and sim and pseudo-sim images_real
    base = os.path.join(pathlib.Path(__file__).parent.absolute(), "alligned_sim_real_tub")
    ri_path, pri_path = os.path.join(base, "real"), os.path.join(base, "pseudoreal")
    si_path, psi_path = os.path.join(base, "sim"), os.path.join(base, "pseudosim")

    # Getting SSIM scores
    real_scores = get_ssim_scores(ri_path, pri_path)
    sim_scores = get_ssim_scores(si_path, psi_path)

    # Printing statistics about scores
    print("\n\n\t\tReal and pseudo-real scores:")
    print_stats(real_scores)
    print("\n\n\t\tSim and pseudo-sim scores:")
    print_stats(sim_scores)


if __name__ == '__main__':
    main()
