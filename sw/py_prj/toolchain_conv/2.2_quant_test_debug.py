import numpy as np
import util

conv1_weight = np.load("./py_prj/output_conv/py_output/2_conv1_weight_q.npy")
conv2_weight = np.load("./py_prj/output_conv/py_output/2_conv2_weight_q.npy")
fc1_weight = np.load("./py_prj/output_conv/py_output/2_fc1_weight_q.npy")

test_data = util.load_data()

OUTPUT_CN = 8
KERNEL_HW = 3

cnt1 = 0
PIC_NUM = 10
for index in range(PIC_NUM):
    input = test_data[index,:-1]
    input = input.reshape(28,28)

    conv1_out = np.zeros((OUTPUT_CN,26, 26))

    for c in range(OUTPUT_CN):
        for i in range(26):
            for j in range(26):
                for k in range(KERNEL_HW):
                    for l in range(KERNEL_HW):
                        # conv1_out[c, i, j] += int(input[ i+k, j+l]) * conv1_weight[c, 0, k, l]
                        conv1_out[c, i, j] += int(input[ i+k, j+l] / 2) * conv1_weight[c, 0, k, l]

    # print(conv1_out[0][5])

    conv1_out = np.maximum(0, conv1_out)/256

    pool1_out = np.zeros((OUTPUT_CN,int(26/2),int(26/2)))
    for c in range(OUTPUT_CN):
        for i in range(int(26/2)):
            for j in range(int(26/2)):
                pool1_out[c, i, j] += conv1_out[c, i*2:i*2+2, j*2:j*2+2].max()        
            pass

    # print(pool1_out[0][2])

    conv2_out = np.zeros((OUTPUT_CN,int(22/2),int(22/2)))
    for n in range(OUTPUT_CN):
        for c in range(OUTPUT_CN):
            for i in range(int(22/2)):
                for j in range(int(22/2)):
                    for k in range(KERNEL_HW):
                        for l in range(KERNEL_HW):
                            conv2_out[n, i, j] += (pool1_out[c, i+k, j+l] * conv2_weight[n, c, k, l])
                        pass
    
    # print(conv2_out[0][2])
    conv2_out = np.maximum(0, conv2_out)/256

    pool2_out = np.zeros((OUTPUT_CN,int(20/4),int(20/4)))
    for c in range(OUTPUT_CN):
        for i in range(int(20/4)):
            for j in range(int(20/4)):
                pool2_out[c, i, j] += conv2_out[c, i*2:i*2+2, j*2:j*2+2].max()        
            pass

    # print(pool2_out[0][2])
    pool2_out = pool2_out.flatten()

    fc1_out = pool2_out.dot(fc1_weight.T)
    # print(fc1_out)
    numpy_out = np.maximum(0, fc1_out)
    pred_labels_numpy = np.argmax(numpy_out)
    print(pred_labels_numpy)
    if pred_labels_numpy == test_data[index][784]:
        cnt1 += 1
    if(index%200==0 and index != 0):
        print(cnt1/(index+1)*100)
print("after quant  acc: %.2f %% " %(cnt1/PIC_NUM*100))


