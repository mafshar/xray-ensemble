import os
import glob
import sys

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return
