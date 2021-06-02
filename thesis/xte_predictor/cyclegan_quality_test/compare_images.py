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
