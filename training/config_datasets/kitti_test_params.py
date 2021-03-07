from dotmap import DotMap

kitti_test_params = DotMap()
kitti_test_params.is_train_dataset = False
kitti_test_params.batch_size = 1
kitti_test_params.pathfile = '/home/khalil/Documents/depth/data/kitti/val.txt'
kitti_test_params.input_shape = (352, 704, 3)
kitti_test_params.img_path = '/home/khalil/Documents/depth/data/kitti/img'
kitti_test_params.depth_path = '/home/khalil/Documents/depth/data/kitti/depth'