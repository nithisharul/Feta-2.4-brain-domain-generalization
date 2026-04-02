import argparse
import os
import SimpleITK as sitk
from utils import get_file, save_acknowledgements


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Given a BIDS formatted `feta_dir` and a `biometry_dir` maps the images to the transformed"
            " space and saves the results in the `out_dir`."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--feta_dir",
        help="Path to the original FeTA images directory.",
        required=True,
    )

    parser.add_argument(
        "--biometry_dir",
        help="Path to the biometric measurements directory.",
        required=True,
    )

    parser.add_argument(
        "--out_dir",
        help="Path to the output directory.",
        required=True,
    )

    args = parser.parse_args()

    feta_dir = os.path.abspath(args.feta_dir)
    biometry_dir = os.path.abspath(args.biometry_dir)
    out_dir = os.path.abspath(args.out_dir)

    # Check that the folders exist
    if not os.path.exists(feta_dir):
        raise FileNotFoundError(f"Folder {feta_dir} does not exist.")
    if not os.path.exists(biometry_dir):
        raise FileNotFoundError(f"Folder {biometry_dir} does not exist.")

    os.makedirs(out_dir, exist_ok=True)
    save_acknowledgements(out_dir, name_suffix="Realigned images")
    sub_list = sorted([f[4:] for f in os.listdir(feta_dir) if "sub" in f])

    for sub in sub_list:
        sub_path_feta = os.path.join(feta_dir, f"sub-{sub}/anat/")
        sub_path_bio = os.path.join(biometry_dir, f"sub-{sub}/anat/")
        if not os.path.exists(sub_path_bio):
            print(f"No biometry for subject {sub}.")
            continue

        imp = get_file(sub_path_feta, suffix="T2w.nii.gz")
        im = sitk.ReadImage(imp)
        trf = sitk.ReadTransform(get_file(sub_path_feta, suffix=".txt"))

        out_path = os.path.join(
            out_dir, f"sub-{sub}/anat/{os.path.basename(imp)}"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im = sitk.Resample(im, im, trf, sitk.sitkLinear)
        sitk.WriteImage(im, out_path)


if __name__ == "__main__":
    main()
