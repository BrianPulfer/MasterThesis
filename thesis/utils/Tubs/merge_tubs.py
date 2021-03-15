import os
import shutil
import argparse


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tub1", type=str, help="Path to source tub320x240_train 1.")
    parser.add_argument("--tub2", type=str, help="Path to source tub320x240_train 2.")
    parser.add_argument("--output", type=str, help="Path to output tub320x240_train.")
    args = dict(vars(parser.parse_args()))

    tub1_path = args['tub1']
    tub2_path = args['tub2']
    tub_new_path = args['output']

    if not tub1_path or not tub2_path or not tub_new_path:
        raise RuntimeError("Usage: --tub1 <path/to/tub1> --tub2 <path/to/tub2> --output <path/to/output>")

    if not os.path.isdir(tub1_path) or not os.path.isdir(tub2_path) or not os.path.isdir(tub_new_path):
        print("One of the specified tub320x240_train paths is invalid!")
    else:
        print("OK! All paths exist.")

    # Copying tub1 directory to destination
    copytree(tub1_path, tub_new_path)

    # Highest record number for tub320x240_train 1
    max_num_tub1 = max([int(filename.split('_')[0]) for filename in os.listdir(tub1_path) if "cam" in filename])

    # Moving old tub320x240_train to new one
    counter = 1
    for filename in sorted(os.listdir(tub2_path)):
        if "cam" not in filename:
            continue

        # Getting new names
        new_nr = str(max_num_tub1 + counter)
        counter += 1

        old_name_cam = filename
        old_name_record = 'record_' + filename.split("_")[0] + ".json"
        new_name_cam = new_nr + '_' + '_'.join(filename.split('_')[1:])
        new_name_record = "record_" + new_nr + ".json"

        # Moving the camera image
        src, dst = os.path.join(tub2_path, old_name_cam), os.path.join(tub_new_path, new_name_cam)
        shutil.move(src, dst)

        # Moving the JSON record
        src, dst = os.path.join(tub2_path, old_name_record), os.path.join(tub_new_path, new_name_record)
        shutil.move(src, dst)


if __name__ == '__main__':
    main()
