import os


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
