from dotmap import DotMap

todd_train_params = DotMap()
todd_train_params.is_train_dataset = True
todd_train_params.batch_size = 4
todd_train_params.input_shape = (352, 704, 3)
todd_train_params.path = '/home/khalil/Documents/depth/data/todd/data'
todd_train_params.locations = ['campbell','cupertino','losgatos', 'paloalto', 'saratoga']
# todd_train_params.locations = ['campbell','cupertino','losgatos', 'paloalto', 'saratoga']