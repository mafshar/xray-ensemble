import os
import glob
import sys

from shutil import move, copyfile

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return

def mv(src, dst):
    try:
        move(src, dst)
    except:
        print 'File', src, 'does not exist'
    return

def cp(src, dst):
    try:
        copyfile(src, dst)
    except:
        print 'File', src, 'does not exist'
    return

def read_textfile_by_line(file):
    content = None
    with open(file) as fh:
        content = fh.readlines()

    if content:
        content = map(lambda x: x.strip(), content)
    return content
