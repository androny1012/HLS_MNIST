import numpy as np
import util

def quant_relu(input,scale,shift,zp):
    output = np.minimum(2**SAT_BIT1-1, input) # 18-16bit sat
    output = np.maximum(-2**SAT_BIT1, output) # 18-16bit sat
    output = output * scale
    output = output / (2 ** ( 15 + shift))
    output = np.minimum(2**7-1, output)  # 16-8bit sat
    output = np.maximum(-2**7, output)  # 16-8bit sat
    output = np.around(output)
    output = output + zp
    output = np.maximum(zp, output)

    return output

conv1_weight = np.load("./py_prj/output_conv/py_output/2_conv1_weight_q.npy")
conv2_weight = np.load("./py_prj/output_conv/py_output/2_conv2_weight_q.npy")
fc1_weight = np.load("./py_prj/output_conv/py_output/2_fc1_weight_q.npy")

test_data = util.load_data()

OUTPUT_CN = 8
KERNEL_HW = 3

cnt1 = 0
PIC_NUM = 1

SAT_BIT1 = 16
for index in range(PIC_NUM):
    input = test_data[index,:-1]
    input = input.reshape(28,28)

    conv1_scale = 20697
    conv1_shift = 9
    conv1_zp    = 46
    
    conv1_out = np.zeros((OUTPUT_CN,26, 26))
    for c in range(OUTPUT_CN):
        for i in range(26):
            for j in range(26):
                for k in range(KERNEL_HW):
                    for l in range(KERNEL_HW):
                        conv1_out[c, i, j] += input[ i+k, j+l] * conv1_weight[c, 0, k, l] / 2
                    pass


    conv1_out = quant_relu(conv1_out,conv1_scale,conv1_shift,conv1_zp)

    # np.save('./py_prj/output_hw/conv1_out_gold',conv1_out)

    pool1_out = np.zeros((OUTPUT_CN,int(26/2),int(26/2)))
    for c in range(OUTPUT_CN):
        for i in range(int(26/2)):
            for j in range(int(26/2)):
                pool1_out[c, i, j] += conv1_out[c, i*2:i*2+2, j*2:j*2+2].max()        
            pass

    # np.save('./py_prj/output_hw/pool1_out_gold',pool1_out)

    conv2_scale = 28089
    conv2_shift = 9
    conv2_zp = 84

    pool1_out = pool1_out - conv1_zp

    conv2_out = np.zeros((OUTPUT_CN,int(22/2),int(22/2)))
    for n in range(OUTPUT_CN):
        for c in range(OUTPUT_CN):
            for i in range(int(22/2)):
                for j in range(int(22/2)):
                    for k in range(KERNEL_HW):
                        for l in range(KERNEL_HW):
                            conv2_out[n, i, j] += (pool1_out[c, i+k, j+l] * conv2_weight[n, c, k, l])
                        pass

    conv2_out = quant_relu(conv2_out,conv2_scale,conv2_shift,conv2_zp)

    # np.save('./py_prj/output_hw/conv2_out_gold',conv2_out)

    pool2_out = np.zeros((OUTPUT_CN,int(20/4),int(20/4)))
    for c in range(OUTPUT_CN):
        for i in range(int(20/4)):
            for j in range(int(20/4)):
                pool2_out[c, i, j] += conv2_out[c, i*2:i*2+2, j*2:j*2+2].max()        
            pass

    # np.save('./py_prj/output_hw/pool2_out_gold',pool2_out)
    
    pool2_out = pool2_out.flatten()

    pool2_out = pool2_out - conv2_zp

    fc1_scale = 16952
    fc1_shift = 7
    fc1_zp = 75
    
    fc1_out = pool2_out.dot(fc1_weight.T)

    fc1_out = quant_relu(fc1_out,fc1_scale,fc1_shift,fc1_zp)

    pred_labels_numpy = np.argmax(fc1_out)
    if pred_labels_numpy == test_data[index][784]:
        cnt1 += 1
    if(index % 200 == 0 and index != 0):
        print(cnt1/(index+1)*100)
        pass

print("numpy  acc: %.2f %% " %(cnt1/PIC_NUM*100))
# 86.12
# 88.61

