from dotmap import DotMap

nyu_test_params = DotMap()
nyu_test_params.is_train_dataset = False
nyu_test_params.batch_size = 1
nyu_test_params.pathfile = '/home/khalil/Documents/depth/data/nyu/val.txt'
nyu_test_params.input_shape = (416, 544, 3)
nyu_test_params.img_path = '/home/khalil/Documents/depth/data/nyu/nyudepthv2'
nyu_test_params.depth_path = '/home/khalil/Documents/depth/data/nyu/nyudepthv2'