# JRDB for Prediction Dataset Setup

## Get the JRDB Dataset

1. Go to https://jrdb.erc.monash.edu/#downloads
2. Create a User or login.
3. Download and extract `JRDB 2022 Full Train Dataset` to `<data_path>/train_dataset`.
4. Download and extract `JRDB 2022 Full Test Dataset` to `<data_path>/test_dataset`.
5. Download and extract `Train Detections` from the JRDB 2019 section to `<data_path>/detections`.

## Get the Leaderboard Test Set Tracks
Download and extract this leaderboard  [3D tracking result](https://jrdb.erc.monash.edu/leaderboards/download/1605) to `<data_path>/test_dataset/labels/raw_leaderboard/`. Such that you have `<data_path>/test_dataset/labels/raw_leaderboard/00XX.txt` This is the best available leaderboard tracker at the time the code was developed.

## Get the Robot Odometry Preprocessed Keypoints

Download the compressed data file [here](https://storage.googleapis.com/gresearch/human_scene_transformer/data.zip).

Extract the files and move them to `<data_path>/processed/` such that you have `<data_path>/processed/odoemtry_train`,  `<data_path>/processed/odoemtry_test` and `<data_path>/processed/labels/labels_3d_keypoints_train/`, `<data_path>/processed/labels/labels_3d_keypoints_test/`.

## Create Real-World Tracks for Test Data

Adapt `<data_path>` in `jrdb_train_detections_to_tracks.py`

Then run

```python jrdb_train_detections_to_tracks.py```

## Dataset Folder

You should end up with a dataset folder of the following structure

```
- <data_path>
  - train_dataset
    - calibration
    - detections
    - images
    - labels
    - pointclouds
  - test_dataset
    - calibration
    - images
    - labels
    - pointclouds
  - processed
    - labels
      - labels_3d_keypoints_test
      - labels_3d_keypoints_train
      - labels_detections_3d
    - odoemtry_test
    - odoemetry_train
```

## Generate the Tensorflow Dataset
Adapt `<data_path>` in `jrdb_preprocess_train.py` and `jrdb_preprocess_test.py`.

Set `<output_path>` in `jrdb_preprocess_train.py` and `jrdb_preprocess_test.py` to where you want to store the processed tensorflow dataset.

```python jrdb_preprocess_train.py```

```python jrdb_preprocess_test.py```