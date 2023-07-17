import os
import numpy as np
import struct


MNIST_DIR = "./dataset/mnist/MNIST/raw"
TEST_DATA = "t10k-images-idx3-ubyte"
TEST_LABEL = "t10k-labels-idx1-ubyte"

def load_mnist(file_dir, is_images = 'True'):
    # Read binary data
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    # Analysis file header
    if is_images:
        # Read images
        fmt_header = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, 0)
    else:
        # Read labels
        fmt_header = '>ii'
        magic, num_images = struct.unpack_from(fmt_header, bin_data, 0)
        num_rows, num_cols = 1, 1
    data_size = num_images * num_rows * num_cols
    mat_data = struct.unpack_from('>' + str(data_size) + 'B', bin_data, struct.calcsize(fmt_header))
    mat_data = np.reshape(mat_data, [num_images, num_rows * num_cols])
    return mat_data

def load_data():
    test_images = load_mnist(os.path.join(MNIST_DIR, TEST_DATA), True)
    test_labels = load_mnist(os.path.join(MNIST_DIR, TEST_LABEL), False)
    test_data = np.append(test_images, test_labels, axis=1)
    return test_data

def createFile(filePath):
    if os.path.exists(filePath):
        print('%s:存在'%filePath)
    else:
        try:
            os.mkdir(filePath)
            print('新建文件夹：%s'%filePath)
        except Exception as e:
            os.makedirs(filePath)
            print('新建多层文件夹：%s' % filePath)