import os
import torch
import torchvision
import pathlib
from fid_score.fid_score import FidScore


def get_frechet_inception_distance(tub1, tub2):
    return FidScore([tub1, tub2]).calculate_fid_score()


def main():
    # Defining path to real and pseudo-real images_real and sim and pseudo-sim images_real
    base = os.path.join(pathlib.Path(__file__).parent.absolute(), "alligned_sim_real_tub")
    ri_path, pri_path = os.path.join(base, "real"), os.path.join(base, "pseudoreal")
    si_path, psi_path = os.path.join(base, "sim"), os.path.join(base, "pseudosim")

    # Getting Frechet Inception Distance
    real_fid = get_frechet_inception_distance(ri_path, pri_path)
    sim_fid = get_frechet_inception_distance(si_path, psi_path)

    # Printing FID
    print(f"Real FID: {real_fid}")
    print(f"Simulated FID: {sim_fid}")


if __name__ == '__main__':
    main()


def scroll_through_images():
    import cv2

    base = os.path.join(pathlib.Path(__file__).parent.absolute(), "alligned_sim_real_tub")
    ri_path, pri_path = os.path.join(base, "real"), os.path.join(base, "pseudoreal")
    si_path, psi_path = os.path.join(base, "sim"), os.path.join(base, "pseudosim")

    ri_names = sorted(list(os.listdir(ri_path)), key=lambda name: int(name.split("_")[0]))
    pri_names = sorted(list(os.listdir(pri_path)), key=lambda name: int(name.split("_")[1]))
    si_names = sorted(list(os.listdir(si_path)), key=lambda name: int(name.split("_")[1].split(".")[0]))
    psi_names = sorted(list(os.listdir(psi_path)), key=lambda name: int(name.split("_")[0]))

    for r, s, pr, ps in zip(ri_names, si_names, pri_names, psi_names):
        real = cv2.resize(cv2.imread(os.path.join(ri_path, r)), (256, 256))
        sim = cv2.resize(cv2.imread(os.path.join(si_path, s)), (256, 256))
        preal = cv2.imread(os.path.join(pri_path, pr))
        psim = cv2.imread(os.path.join(psi_path, ps))

        cv2.imshow("real", real)
        cv2.imshow("sim", sim)
        cv2.imshow("pseudo-real", preal)
        cv2.imshow("pseudo-sim", psim)

        key = cv2.waitKey()

        # Press 'S' to store images
        if key == 115:
            cv2.imwrite(os.path.join(base, "real.jpg"), real)
            cv2.imwrite(os.path.join(base, "sim.jpg"), sim)
            cv2.imwrite(os.path.join(base, "preal.png"), preal)
            cv2.imwrite(os.path.join(base, "psim.png"), psim)
