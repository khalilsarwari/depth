from dotmap import DotMap

nyu_train_params = DotMap()
nyu_train_params.is_train_dataset = True
nyu_train_params.batch_size = 16
nyu_train_params.pathfile = '/home/khalil/Documents/depth/data/nyu/train.txt'
nyu_train_params.input_shape = (416, 544, 3)
nyu_train_params.img_path = '/home/khalil/Documents/depth/data/nyu/nyudepthv2'
nyu_train_params.depth_path = '/home/khalil/Documents/depth/data/nyu/nyudepthv2'