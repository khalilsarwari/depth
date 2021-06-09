# Depth

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
    waymo/
      waymo_open_dataset_v_1_2_0/
        training/
          segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
        validation/
          ...
    nyu/
        train.txt
        val.txt
        train/
            bathroom/
                rgb_00043.jpg
                sync_depth_00043.png
                ...
        test/
            bathroom/
                rgb_00045.jpg
                sync_depth_00045.png
                ...
```

# Pretrained Models
Pretrained models can be downloaded [here](https://drive.google.com/drive/folders/1bJ0TH2E_Cl5HKxTum9ajKWPomBIVvd0y?usp=sharing)

# Collecting Data

Sometimes the camera does not initialize correctly, in which case you need to run the following command to reset the usb device:
```
cc usbreset.c -o usbreset && chmod a+x usbreset && sudo ./usbreset /dev/bus/usb/002/003
```

Use `lsusb` to find the appropriate bus/device number

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
