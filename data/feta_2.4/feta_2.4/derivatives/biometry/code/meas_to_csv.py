import argparse
import os
import SimpleITK as sitk
import pandas as pd
from utils import get_file, get_dist, REGION_DICT


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Given a BIDS formatted `biometry_dir` containing biometric measurement masks,"
            "  computes the distance between the measurements and saves them in a csv file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--biometry_dir",
        help="Path to the biometric measurements directory.",
        required=True,
    )

    parser.add_argument(
        "--out_csv",
        help="Path to the output csv file.",
        required=True,
    )

    args = parser.parse_args()

    biometry_dir = os.path.abspath(args.biometry_dir)

    sub_list = sorted([f[4:] for f in os.listdir(biometry_dir) if "sub" in f])
    out_dir = os.path.dirname(args.out_csv)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(columns=["sub", "LCC", "HV", "bBIP", "sBIP", "TCD"])

    for i, sub in enumerate(sub_list):
        print(f"Processing subject {sub} ({i+1}/{len(sub_list)})")
        sub_path_bio = os.path.join(biometry_dir, f"sub-{sub}/anat/")
        mask = sitk.ReadImage(get_file(sub_path_bio, suffix="meas.nii.gz"))
        df.loc[i, "sub"] = f"sub-{sub}"
        for region in REGION_DICT.keys():
            df.loc[i, region] = get_dist(mask, region)
        print(df.loc[i].T)
    df.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
