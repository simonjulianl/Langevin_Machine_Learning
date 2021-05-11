import socket
import os
import gc
import torch.cuda

def collect_garbages():
    gc.collect()
    ngarbage = len(gc.garbage)
    print('collected garbage, uncollectible amount ',ngarbage)

def print_uncollectible_garbages():
    print('uncollectible garbages are')
    for g in gc.garbage:
        print(g)

def print_system_logs(device):

    print('OS process id ',os.getpid())
    print(socket.gethostname())
    print(device)

    if torch.cuda.is_available():
        print('cur cuda dev ',torch.cuda.current_device())
        print('total gpu available ',torch.cuda.device_count())

