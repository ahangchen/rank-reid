import os


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content)