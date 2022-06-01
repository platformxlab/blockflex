import datetime as dt
import sys
import subprocess
import os
import argparse
import struct
import codecs
import time
from collections import defaultdict
import numpy as np
# from colorama import Fore, Style

KB = 1024
MB = 1024**2
GB = 1024**3

# Directories
USR_DIR="/home/osdi22ae/BlockFlex/ocssd" # CHANGE THIS PATH 
BLOCKFLEX_DIR="/home/osdi22ae/BlockFlex/blockflex" # CHANGE THIS PATH
PFE_USR_DIR = os.path.join(USR_DIR, "iscsi") 
IMG_DIR=os.path.join(USR_DIR, "imgs")
PFE_LOG_DIR=os.path.join(USR_DIR, "logs")

# Logs
TGTD_LOG=os.path.join(PFE_LOG_DIR, "log.tgtd")
IO_HEAD_LOG=os.path.join(PFE_LOG_DIR, "pfe_io_head_log")
IO_DATA_LOG=os.path.join(PFE_LOG_DIR, "pfe_io_data_log")
IO_HEAD_SPLIT_LOG=os.path.join(PFE_LOG_DIR, "pfe_io_head_split_log")
ERR_BLOCK_LOG=os.path.join(PFE_LOG_DIR, "pfe_err_block_log")

# Targets
ISCSI_TARGET_IP="127.0.0.1"
ISCSI_TARGET_NAMES=[f"harvest_target:{id}" for id in range(10)]
ISCSI_TARGET_IMGS=[os.path.join(IMG_DIR, f"dummy_disk_{id}.img") for id in range(10)]
DUMMY_DISK_SIZE=2*GB # dummy backstore files are 2GB 
DUMMY_DISK_SIZE_STRING="1T" # we present the virtual disk as 1TB

# Alias
esudo = "sudo env \"PATH=$PATH\""

# Flags
HARVEST_FLAG = f"{BLOCKFLEX_DIR}/start_harvest"
END_FLAG = f"{BLOCKFLEX_DIR}/end_harvest"
QUEUES = ["/dev/mqueue/harvest0", "/dev/mqueue/harvest1"]

DEBUG = True
VERBOSE = True

def log_msg(*msg, out_f=None, emphasize=False, warning=False):
    '''
    Log a message with the current time stamp.
    '''
    msg = [str(_) for _ in msg]
    out_str = "[%s] %s" % ((dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), " ".join(msg))
    if emphasize:
        out_str = bcolors.OKGREEN + out_str + bcolors.ENDC
    if warning:
        out_str = bcolors.WARNING + out_str + bcolors.ENDC

    if DEBUG:
        if out_f is not None:
            print(out_str, file=out_f)
            out_f.flush()
        else:
            print(out_str)

def run_command(command_str, cwd=None, output=False, error=False, verbose=VERBOSE, suppress_output=False):
    if verbose:
        log_msg(command_str)

    f_args = {}

    # if not debug:
    #     f_args['stdout'] = subprocess.DEVNULL

    if cwd:
        f_args['cwd'] = cwd
    
    if output:
        f_args['stdout']=subprocess.PIPE

    if suppress_output:
        f_args['stdout']=subprocess.DEVNULL
    
    if error:
        f_args['stderr']=subprocess.PIPE

    result = subprocess.run(command_str, shell=True, **f_args)

    ret = dict()
    if output:
        ret['stdout'] = result.stdout.decode("utf-8")

    if error:
        ret['stderr'] = result.stderr.decode("utf-8")
    
    return ret

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'