from dotmap import DotMap

kitti_train_params = DotMap()
kitti_train_params.is_train_dataset = True
kitti_train_params.batch_size = 16
kitti_train_params.pathfile = '/home/khalil/Documents/depth/data/kitti/train.txt'
kitti_train_params.input_shape = (352, 704, 3)
kitti_train_params.img_path = '/home/khalil/Documents/depth/data/kitti/img'
kitti_train_params.depth_path = '/home/khalil/Documents/depth/data/kitti/depth'