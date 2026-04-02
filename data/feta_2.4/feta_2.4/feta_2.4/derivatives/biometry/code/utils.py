import os
import json
import numpy as np
import SimpleITK as sitk

ACKNOWLEDGEMENTS = {
    "Name": "Fetal Tissue Annotation Challenge FeTA Dataset",
    "BIDSVersion": "XXX",
    "DatasetType": "derivative",
    "License": "XXX",
    "Authors": ["Thomas Sanchez, Meritxell Bach Cuadra"],
    "Acknowledgements": "Special thanks to the annotators who executed the biometric meeasurements, Yvan Gomez, Andras Jakab and Mériam Koob",
    "HowToAcknowledge": "???",
    "Funding": ["???"],
    "EthicsApprovals": "???",
}
REGION_DICT = {"LCC": 1, "HV": 2, "bBIP": 3, "sBIP": 4, "TCD": 5}


def get_dist(im, region):
    """
    Get the distance between the two points of a given region.
    """
    x, y, z = np.where(sitk.GetArrayFromImage(im) == REGION_DICT[region])
    if len(x) == 0:
        return np.nan
    p1 = np.array([x[0], y[0], z[0]])
    p2 = np.array([x[1], y[1], z[1]])
    ip_res = im.GetSpacing()[0]
    assert len(x) == 2, f"Region {region} has {len(x)} points"
    dist = round(np.linalg.norm((p1 - p2) * ip_res), 2)
    return dist


def transform_measurements(biometry_mask, trf):
    """
    Transform the biometry mask to the original image space. This transform
    is applied to each individual point of the biometry mask.
    """
    biometry_mask_og = sitk.GetImageFromArray(
        np.zeros_like(sitk.GetArrayFromImage(biometry_mask))
    )
    biometry_mask_og.CopyInformation(biometry_mask)
    for z, y, x in zip(*np.where(sitk.GetArrayFromImage(biometry_mask) > 0)):
        z, y, x = int(z), int(y), int(x)
        xm, ym, zm = biometry_mask.TransformPhysicalPointToIndex(
            trf.TransformPoint(
                biometry_mask.TransformContinuousIndexToPhysicalPoint(
                    [x, y, z]
                )
            )
        )
        biometry_mask_og[xm, ym, zm] = biometry_mask[x, y, z]
    return biometry_mask_og


def save_acknowledgements(out_dir, name_suffix=""):
    """
    Save the acknowledgements in the out_dir.
    """
    ACKNOWLEDGEMENTS["Name"] += f" - {name_suffix}"
    with open(os.path.join(out_dir, "dataset_description.json"), "w") as f:
        json.dump(ACKNOWLEDGEMENTS, f, indent=4)


def get_file(folder, suffix="T2w.nii.gz"):
    """
    Get a file ending with a given suffix in a given folder
    and return it. If multiple files are found, return the first one.
    """
    for file in os.listdir(folder):
        if file.endswith(suffix):
            return os.path.join(folder, file)
    return None
