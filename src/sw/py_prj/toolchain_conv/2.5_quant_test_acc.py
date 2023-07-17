# coding:utf-8
from conv_acc.layers_1 import FullyConnectedLayer
from conv_acc.layers_2 import ConvolutionalLayer, MaxPoolingLayer
import numpy as np
import util
from tqdm import tqdm
import time

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

def test(PIC_NUM):

    conv1_weight = np.load("./py_prj/output_conv/py_output/2_conv1_weight_q.npy")
    conv2_weight = np.load("./py_prj/output_conv/py_output/2_conv2_weight_q.npy")
    fc1_weight = np.load("./py_prj/output_conv/py_output/2_fc1_weight_q.npy")

    OUTPUT_CN = 8
    KERNEL_HW = 3

    cnt1 = 0

    desc = "without acc                       "
    for index in tqdm(range(PIC_NUM), desc=desc):
        input = test_data[index,:-1]
        input = input.reshape(28,28)

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
        fc1_out = pool2_out.dot(fc1_weight.T)
        fc1_out = quant_relu(fc1_out,fc1_scale,fc1_shift,fc1_zp)

        pred_labels_numpy = np.argmax(fc1_out)
        if pred_labels_numpy == test_data[index][784]:
            cnt1 += 1

    print("numpy  acc: %.2f %% " %(cnt1/PIC_NUM*100))

def test_speed_up(PIC_NUM, quan_type = 0):

    conv1_layer = ConvolutionalLayer(3, 1, 8, 0, 1, 1)  #k,ic,oc.paddding,stride,speedUP
    pool1_layer = MaxPoolingLayer(2, 2, 1)
    conv2_layer = ConvolutionalLayer(3, 8, 8, 0, 1, 1)
    pool2_layer = MaxPoolingLayer(2, 2, 1)
    fc1_layer = FullyConnectedLayer(200,10)


    conv1_weight = np.load("./py_prj/output_conv/py_output/2_conv1_weight_q.npy")
    conv2_weight = np.load("./py_prj/output_conv/py_output/2_conv2_weight_q.npy")
    fc1_weight = np.load("./py_prj/output_conv/py_output/2_fc1_weight_q.npy")

    conv1_weight = np.transpose(conv1_weight,(1,2,3,0))
    conv2_weight = np.transpose(conv2_weight,(1,2,3,0))
    fc1_weight = fc1_weight.T

    # conv1_weight = np.random.rand(1, 3, 3, 8)  # ic,k,k,oc
    conv1_bias = np.zeros(8)
    conv1_layer.init_param()
    conv1_layer.load_param(conv1_weight, conv1_bias)
    
    # conv2_weight = np.random.rand(8, 3, 3, 8) # ic,k,k,oc
    conv2_bias = np.zeros(8)
    conv2_layer.init_param()
    conv2_layer.load_param(conv2_weight, conv2_bias)

    # fc1_weight = np.random.rand(200,10)
    fc1_bias = np.zeros([1,10])
    fc1_layer.init_param()
    fc1_layer.load_param(fc1_weight, fc1_bias)

    correct_cnt = 0

    if(quan_type == 0):
        desc = "quant with scale and zero point   "
    else:
        desc = "quant without scale and zero point"
    # for index in range(PIC_NUM):
    for index in tqdm(range(PIC_NUM), desc=desc):
        input = test_data[index,:-1]
        label = test_data[index,784]

        input = input.reshape(28,28)
        input_data = np.random.rand(1, 1, 28, 28)
        input_data[0,0,:,:] = input

        if(quan_type == 0):
            conv1_out = conv1_layer.forward(input_data/2)
            conv1_out = quant_relu(conv1_out,conv1_scale,conv1_shift,conv1_zp)
            pool1_out = pool1_layer.forward(conv1_out) - conv1_zp
            conv2_out = conv2_layer.forward(pool1_out)
            conv2_out = quant_relu(conv2_out,conv2_scale,conv2_shift,conv2_zp)
            pool2_out = pool2_layer.forward(conv2_out) - conv2_zp
            pool2_out_flatten = pool2_out.flatten()
            fc1_out = fc1_layer.forward(pool2_out_flatten)
            fc1_out = quant_relu(fc1_out,fc1_scale,fc1_shift,fc1_zp)
            pred = np.argmax(fc1_out)
        else:
            conv1_out = conv1_layer.forward(input_data/2) # /2 for int8
            conv1_out = np.maximum(0, conv1_out)/256
            pool1_out = pool1_layer.forward(conv1_out)
            conv2_out = conv2_layer.forward(pool1_out)
            conv2_out = np.maximum(0, conv2_out)/256
            pool2_out = pool2_layer.forward(conv2_out)
            pool2_out_flatten = pool2_out.flatten()
            fc1_out = fc1_layer.forward(pool2_out_flatten)
            pred = np.argmax(fc1_out)            
        # print(label,pred)
        if pred == label:
            correct_cnt += 1

    if(quan_type == 0):
        print("quant with scale and zero point,    acc: %.2f %% " %(correct_cnt/PIC_NUM*100))
    else:
        print("quant without scale and zero point, acc: %.2f %% " %(correct_cnt/PIC_NUM*100))
if __name__ == '__main__':
    test_data = util.load_data()

    SAT_BIT1 = 16

    conv1_scale = 20697
    conv1_shift = 9
    conv1_zp    = 46

    conv2_scale = 28089
    conv2_shift = 9
    conv2_zp = 84

    fc1_scale = 16952
    fc1_shift = 7
    fc1_zp = 75
    
    PIC_NUM = 10000
    test_speed_up(PIC_NUM,0)
    # stamp = time.time()
    # test(PIC_NUM) # 很慢，不要用大量数据测试
    # conv_forward_time = time.time()-stamp

    # stamp = time.time()
    # test_speed_up(PIC_NUM,0)
    # speedup_conv_forward_time = time.time()-stamp
    # # test_speed_up(10,1)


    # print('conv forward         time: %f ms'%(conv_forward_time*1000))
    # print('conv forward speedup time: %f ms'%(speedup_conv_forward_time*1000))

    # print('CONV FORWARD SPEEDUP RATIO: %f'%(conv_forward_time / speedup_conv_forward_time))

# quant with scale and zero point,    acc: 88.61 %
# quant without scale and zero point, acc: 87.43 %
# conv forward         time: 727985.103369 ms
# conv forward speedup time: 14625.301838 ms
# CONV FORWARD SPEEDUP RATIO: 49.775732