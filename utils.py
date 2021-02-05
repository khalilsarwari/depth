import os
import numpy as np
import random
import json
import time
import shutil
import signal
import subprocess
import torch
import torch.distributed as dist


def kill_processes_on_port(port):
    process = subprocess.Popen(
        ["lsof", "-i", ":{0}".format(port)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for process in str(stdout.decode("utf-8")).split("\n")[1:]:
        data = [x for x in process.split(" ") if x != '']
        if (len(data) <= 1):
            continue
        os.kill(int(data[1]), signal.SIGKILL)


def initialize_dirs_and_files(args, config):
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    exp_path = os.path.join(args.log_dir, args.exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    else:
        print('Deleting old log {} in 5 sec!'.format(exp_path))
        time.sleep(5)
        shutil.rmtree(exp_path)
        os.mkdir(exp_path)
    config.exp_path = exp_path
    weights_path = os.path.join(config.exp_path, args.weights_dir)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    config.weights_path = weights_path
    filename = os.path.join(exp_path, 'config.json')
    json.dump(config.toDict(), open(filename, 'w'),
              default=lambda o: str(o), indent=4)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def init_processes(rank, fn, tp, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = tp.master_addr
    os.environ['MASTER_PORT'] = tp.master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=torch.cuda.device_count())
    set_seed(tp.seed)
    fn(rank)
