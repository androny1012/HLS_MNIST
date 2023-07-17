import numpy as np
import util

# 量化后权重
conv1_weight = np.load("./py_prj/output_conv/py_output/2_conv1_weight_q.npy")
print(conv1_weight[0][0])
print(conv1_weight[1][0])
conv1_weight = conv1_weight.transpose(2,3,1,0)

print(conv1_weight.shape)

# 直接复制到C语言数组中
for w in range(conv1_weight.shape[0]):
    for h in range(conv1_weight.shape[1]):
        for ci in range(conv1_weight.shape[2]):
            for co in range(conv1_weight.shape[3]):
                print(conv1_weight[w][h][ci][co], end=",")

print("")

# 测试图片，也是作为tb中的初始化数组
test_data = util.load_data()
input = test_data[0,:-1]
print(input.shape)
for i in range(input.shape[0]):
    print(int(input[i]/2), end=",")



# 其他曾的权重

# conv2_weight = np.load("./py_prj/output_conv/py_output/2_conv2_weight_q.npy")

# conv2_weight = conv2_weight.transpose(2,3,1,0)

# print(conv2_weight.shape)

# for w in range(conv2_weight.shape[0]):
#     for h in range(conv2_weight.shape[1]):
#         for ci in range(conv2_weight.shape[2]):
#             for co in range(conv2_weight.shape[3]):
#                 print(conv2_weight[w][h][ci][co], end=",")

# print("")
# print("")


# fc1_weight = np.load("./py_prj/output_conv/py_output/2_fc1_weight_q.npy")
# print(fc1_weight.shape)
# for h in range(fc1_weight.shape[0]):
#     for w in range(fc1_weight.shape[1]):
#         print(fc1_weight[h][w], end=",")

# print("")
# print("")

# print(fc1_weight[0][0])
# print(fc1_weight[0][1])
# print(fc1_weight[1][0])
# print(fc1_weight[1][1])
# print(fc1_weight[9][0])
# print(fc1_weight[9][1])


# print("")
# print("")

# print(test_data[0][784])
# print(test_data[1][784])
# print(test_data[2][784])
# print(test_data[3][784])
# print(test_data[4][784])
# print(test_data[5][784])
# print(test_data[6][784])
# print(test_data[7][784])
# print(test_data[8][784])
# print(test_data[9][784])