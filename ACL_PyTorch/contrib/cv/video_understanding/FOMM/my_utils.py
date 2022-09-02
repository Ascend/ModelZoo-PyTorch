import os


def mkdir(path):
    """
    若目录不村子啊，创建目录，否则不做任何操作
    :param path: 目录路径
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)