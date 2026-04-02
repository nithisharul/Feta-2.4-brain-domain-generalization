# Fetal Tissue Annotation Challenge (FeTA) Biometry - MICCAI 2024

This dataset is the training dataset for the biometry task of FeTA Challenge held at the MICCAI 2024 Conference. This derived dataset contains biometric measurement masks with the keypoints used to calculate five anatomical regions on T2-weighted fetal brain super-resolution reconstructions with the following keypoint labels: 

0. Background
1. Height of the vermis
2. Length of the corpus callosum
3. Brain biparietal diameter
4. Skull biparietal diameter
5. Transverse cerebellar diameter

The data are released along a transform file that can be used to re-align the reconstructed T2 images into the plane that was used to measure the biometry.

The goal of this dataset is to encourage research groups to develop automated biometry methods that are robust across a range of gestational ages, multiple centers, a variety of brain pathologies as well as normally developing fetal brains. 

See [fetachallenge.github.io](fetachallenge.github.io) for more details, and to register for the FeTA Challenge.

> If you use this dataset, please cite our Zenodo data: Sanchez, T., Gomez, Y., Koob, M., Jakab, A. and Bach Cuadra, M. (2024) Fetal Tissue Annotation Challenge (FeTA) Biometry - MICCAI 2024.
[https://doi.org/10.5281/zenodo.11192452](https://doi.org/10.5281/zenodo.11192452).

**Notes.** This dataset does _not_ contain the images used for the FeTA challenge, but only the training annotations for the biometry task of the FeTA 2024 challenge. Regarding the access to the image data, please check [this link](https://fetachallenge.github.io/pages/Data_download). The biometry measurements will also be distributed along with the updated version of the dataset from the University Children’s Hospital Zurich on [Synapse](https://www.synapse.org/#!Synapse:syn25649159/wiki/610007) and the Vienna data (to be used for the purpose of the challenge only).

## Detailed information

### Dataset description
The dataset contains the keypoints and transforms (but no imaging data) of the 80 patients from the University Children’s Hospital Zurich used as training data of the FeTA challenge 2024. Some subjects have missing measurements, and 10 subjects (sub-004, sub-007, sub-008, sub-009, sub-015, sub-017, sub-020, sub-022, sub-023, sub-078) were excluded as the quality did not allow for measurements to be performed.

### Folder structure
The files for the biometry are available in the `derviatives/biometry` folder, which contains the following items.

1. `biometry.tsv`: This TSV file contains the ground truth measurements for each structure of interest (bBIP, sBIP, LCC, HV, TCD) for each subject in the FeTA training set. 
As it was not possible to perform all measurements for all subjects, there are occasional missing values. 
2. For each subject folder `sub-<sub>/anat`, there are the following files:
    - `sub-<sub>_rec-<rec>_meas.nii.gz`: The biometric measurements mask with keypoints in the re-aligned space, which was created by the experts to do the biometric measurements.
    - `sub-<sub>_rec-<rec>_trf.txt`: The transformation mapping the original image to the re-aligned space.
3. A `code` folder with example utility scripts.
   - `im_original_to_realigned.py`: Apply the transformation mapping to map them to the re-aligned space, saving them in the `derivatives/im_reo` folder.
   - `meas_realigned_to_original.py`: Apply the inverse transformation mapping to the measurement mask and map it to the original image space, saving them in the `derivatives/biometry_mapped` folder.
   - `meas_to_csv.py` Given a BIDS formatted folder with biometry masks with suffix `_meas`, compute the measurements from the landmark and saves them in a CSV file `predictions.csv`.
   - *Note.* The code relies on commonly used Python libraries (`SimpleITK`, `argparse`, `pandas`, `numpy`, `json` and `os`).

### Data format
The measurement mask `sub-<sub>_rec-<rec>_meas.nii.gz` contains the location of the two landmarks used for each measurements in the form of a mask. The values correspond to the 5 labels above: `{"LCC": 1, "HV": 2, "bBIP": 3, "sBIP": 4, "TCD": 5}`

### Data preprocessing
Measurements were done by clinicians on manually re-aligned images using ITK-Snap, as described in the protocol available in the `documentation` folder. For each subject, a `.annot` file, containing the measurements in the re-aligned space as well as a `.txt` file containing the transform between the spaces was saved. 

### Important remarks
- The transformation from the re-aligned space to the original space introduces minor (up to 1 pixel) change for the biometric measurement. The values given correspond to the biometric measurements done in the re-aligned space. Transforming the measurements mapped to the original space back to the re-aligned space will *not* yield the exact original pixel locations.
- The landmarks have their ends rounded to integer values. This induces a minor discrepancy with the original biometric measurements which are not rounded. This can affect the results up to 0.45mm.
- **Inter-rater reliability.** Landmark estimation and biometric measurements were performed in each case by a single expert, which implies that they would be subject to inter-rater variability. We estimated an inter-rater variability below 1mm on average (except for sBIP where it is close to 1.4mm) for the biometric measurements. On the landmarks, the variability on the landmark location ranges from 1.7mm for HV to 6mm for sBIP. The landmarks provided are noisier than the biometric measurements.



