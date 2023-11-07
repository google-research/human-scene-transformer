*This is not an officially supported Google product.*

# JRDB for Prediction Dataset Setup

## Get the JRDB Dataset

1. Go to https://jrdb.erc.monash.edu/#downloads
2. Create a User or login.
3. Download and extract [JRDB 2022 Full Train Dataset](https://jrdb.erc.monash.edu/static/downloads/JRDB2022/train_dataset_with_activity/train_dataset_with_activity.zip) to `<data_path>/train_dataset`.
4. Download and extract [JRDB 2022 Full Test Dataset](https://jrdb.erc.monash.edu/static/downloads/JRDB2022/test_dataset_without_labels/jrdb22_test.zip) to `<data_path>/test_dataset`.
5. Download and extract [Train Detections](https://jrdb.erc.monash.edu/static/downloads/train_detections.zip) from the JRDB 2019 section to `<data_path>/detections`.

## Get the Leaderboard Test Set Tracks

### For the JRDB Challenge Dataset
Download and extract this leaderboard  [3D tracking result](https://jrdb.erc.monash.edu/leaderboards/download/1762) to `<data_path>/test_dataset/labels/PiFeNet/`. Such that you have `<data_path>/test_dataset/labels/PiFeNet/00XX.txt`.

### For the Orginal Dataset used in the Paper
Download and extract this leaderboard  [3D tracking result](https://jrdb.erc.monash.edu/leaderboards/download/1605) to `<data_path>/test_dataset/labels/ss3d_mot/`. Such that you have `<data_path>/test_dataset/labels/ss3d_mot/00XX.txt`. This was the best available leaderboard tracker at the time the method was developed.

## Get the Robot Odometry

Download the compressed Odometry data file [here](https://storage.googleapis.com/gresearch/human_scene_transformer/odometry.zip).

Extract the files and move them to `<data_path>/processed/` such that you have `<data_path>/processed/odoemtry/train`,  `<data_path>/processed/odoemtry/test`.

Alternatively you can extract the robot odometry from the raw rosbags yourself via `extract_robot_odometry_from_rosbag.py`.

## Get the Preprocessed Keypoints

Download the compressed Keypoints data file [here](https://storage.googleapis.com/gresearch/human_scene_transformer/keypoints.zip).

Extract the files and move them to `<data_path>/processed/` such that you have  `<data_path>/processed/labels/labels_3d_keypoints/train/`, `<data_path>/processed/labels/labels_3d_keypoints/test/`.

## Create Real-World Tracks for Train Data

Run

```python jrdb_train_detections_to_tracks.py --input_path=<data_path>```

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
      - labels_3d_keypoints
        - train
        - test
      - labels_detections_3d
    - odoemtry
      - train
      - test
```

## Generate the Tensorflow Dataset
### For the JRDB Challenge Dataset
```python jrdb_preprocess_train.py --input_path=<data_path> --output_path=<output_path> --max_distance_to_robot=50.0```

```python jrdb_preprocess_test.py --input_path=<data_path> --output_path=<output_path> --max_distance_to_robot=50.0 --tracking_method=PiFeNet --tracking_confidence_threshold=0.01```

Please note that this can take multiple hours due to the processing of the scene's
pointclouds. If you do not need the pointclouds you can speed up the processing
by passing `--process_pointclouds=False` for both.

### For the Orginal Dataset used in the Paper
```python jrdb_preprocess_train.py --input_path=<data_path> --output_path=<output_path> --max_distance_to_robot=15.0```

```python jrdb_preprocess_test.py --input_path=<data_path> --output_path=<output_path> --max_distance_to_robot=15.0 --tracking_method=ss3d_mot```

Please note that this can take multiple hours due to the processing of the scene's
pointclouds. If you do not need the pointclouds you can speed up the processing
by passing `--process_pointclouds=False` for both.