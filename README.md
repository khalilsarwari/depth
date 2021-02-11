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
- [Waymo Dataset](https://waymo.com/open/download/)

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

# Acknowledgements

This code contains selections from the official [Adabins](https://github.com/shariqfarooq123/AdaBins) repo.
