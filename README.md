# Progress and Proposals: A Case Study of Monocular Depth Estimation

Report: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-32.pdf

# Getting Started
This code was developed using Ubuntu 18.04, PyTorch 1.7.0., and CUDA 11.0.

1. Clone this repository, install the requirements:
```
git clone https://github.com/khalilsarwari/depth.git
python -m pip install requirements.txt
```
2. Download the necessary data:
- [KITTI images](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip)
- [KITTI annotated depth](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip) (Note: `train` and `val` folders should be combined after unzipping to make `depth`)
- [TODD dataset](https://drive.google.com/drive/folders/11TMkt_pd2vhKRXAJx71cewwaFsw9Z_-c?usp=sharing) (ours)

The final data folder structure should look something like:
```
data/
    kitti/
      train.txt
      val.txt
      img/
        2011_09_26/
          2011_09_26_drive_0001_sync/
            image_00/
            ...
      depth/
        2011_09_26_drive_0001_sync/
          proj_depth/
            groundtruth/
            ...
    todd/
      data/
        berkeley/
          000036998_1618097868559354454.png
          000036999_1618097868659674521_depth.png
          000036999_1618097868659674521.npy
          ...
        campbell/
          ...
```
# Camera/LiDAR Setup

To check that the camera and LiDAR are setup up correctly, and visualize the image/pointclouds, run:
```
docker-compose --file vis.yml up
```

This will open an rviz window and display the camera and LiDAR messages being received.

# Camera Calibration
Load images matching the paths `ACSC/calibration_data/**/**/*.png` and run

```
python depth/ACSC/camera_calibration.py
```
This will produce camera intrinsic and distortion parameters, with examples given under `ACSC/calibration_data/1617226905.61/`

# Camera-LiDAR Cross Calibration
This step requires user-involvement to select the ROI (calibration board) using the BEV.
See [ACSC](https://github.com/HViktorTsoi/ACSC) for more details.

```
docker-compose --file collect-calibrate.yml up -d && docker attach depth_main_1
```

Then run

```
docker-compose --file calibrate.yml up
```
This will produce the extrinsic parameters for the camera-LiDAR transformation, with an example given at `ACSC/calibration_data/1617226905.61/parameter/extrinsic`

# Collecting Data
To collect image and pointclouds for a given location, run:

```
LOCATION=<location> docker-compose -f collect-data.yml up
```

Sometimes the camera does not initialize correctly, in which case you need to run the following command to reset the usb device:
```
cc usbreset.c -o usbreset && chmod a+x usbreset && sudo ./usbreset /dev/bus/usb/002/003
```
Use `lsusb` to find the appropriate bus/device number

To run the projection of the pointcloud into the image and postprocess, run:

```
docker-compose -f prepare-data.yml up
```

# Pretrained Models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1bJ0TH2E_Cl5HKxTum9ajKWPomBIVvd0y?usp=sharing)

# Running Experiments

Training commands are of the form:
```
python run_train.py -c <param file>
```

For example, to train AdaBin model:
```
python run_train.py -c original_params
```

By default, training scripts will use all available GPUs. To run on a subset, please set the env var i.e. `CUDA_VISIBLE_DEVICES=0`.

# Notes

If you run into issues like `qt.qpa.screen: QXcbConnection: Could not connect to display :0`

run 

`xhost +`

from outside the container

# Acknowledgements

This code contains selections from the official [Adabins](https://github.com/shariqfarooq123/AdaBins) repo.
[ACSC](https://github.com/HViktorTsoi/ACSC) is used for camera-LiDAR cross-calibration
