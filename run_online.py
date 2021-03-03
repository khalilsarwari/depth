import argparse
import importlib
import utils
import subprocess
import os
from datetime import datetime
import numpy as np
import cv2
import os
import errno
from time import time
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def main(args, config):
    FIFO = '/tmp/myfifo'

    try:
        os.mkfifo(FIFO)
    except OSError as oe:
        if oe.errno != errno.EEXIST:
            raise

    while True:
        print("Opening FIFO...")

        with open(FIFO, 'rb') as fifo:
            print("FIFO opened")
            while True:
                data = fifo.read(1280*720*4)
                image = Image.frombytes('RGB', (1280, 720), data, 'raw')
                cv2.imshow('image', np.array(image))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if len(data) == 0:
                    print("Writer closed")
                    break
                print('Read: "{0}"'.format(len(data)))


# When everything done, release the capture
cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="depth")
    parser.add_argument("-c", "--config", type=str,
                        required=True, help='path to config file')
    parser.add_argument("-exp_name", type=str, help='name of this run')
    parser.add_argument("--log_dir", type=str, default='logs')
    parser.add_argument("--weights_dir", type=str, default='weights')
    parser.add_argument('--tb_port', action="store", type=int,
                        default=6006, help="tensorboard port")

    args = parser.parse_args()

    config = importlib.import_module(
        'config_online.{}'.format(args.config)).config
    if not args.exp_name:
        args.exp_name = args.config + '@' + config.model.__name__ + \
            '@' + str(round(datetime.utcnow().timestamp()))

    main(args, config)
