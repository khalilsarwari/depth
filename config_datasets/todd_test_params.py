from dotmap import DotMap

todd_test_params = DotMap()
todd_test_params.is_train_dataset = False
todd_test_params.batch_size = 1
todd_test_params.input_shape = (352, 704, 3)
todd_test_params.path = '/home/khalil/Documents/depth/data/todd/data'
todd_test_params.locations = ['berkeley']