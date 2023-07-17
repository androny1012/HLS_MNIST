python和c语言实现全连接mnist手写数字识别

文件结构

    -py_prj 主要实现部分，包括网络训练，参数导出与量化，参数导出分为C数组和Verilog版本
        -toolchain 完整流程主要代码
            -1_main.py 搭建(784,10)卷积神经网络并训练，固定了随机数种子，可复现结果；
            -2.0_quant_model.py 使用torch的 静态量化方法，得到int8的模型，并对比量化前后结果，精度只下降了非常少，量化效果比直接量化到-127 到 127 要好很多
            -2.1_get_quant_para.py 根据量化的模型生成权重参数和对应的硬件计时要用的三个参数，scale，shift，zp
            -2.2_quant_test_debug.py 手动实现卷积计算的全过程，即6层循环，用于debug看每层计算结果
            -2.3_quant_test_scale_zp.py 加入了量化部分
            -2.4_quant_test_scale_zp.py 借助torch实现的量化，也能够得到量化的中间结果，但是不太好用，主要也不能一步步拆解计算过程
            -2.5_quant_test_acc.py 和上述的计算过程一模一样，但使用了im2col进行加速，加速能达到50倍，用于测试完整数据集在这种量化方案上精度变化，实测是一模一样，如果硬件比这个低，说明硬件实现错误
            -3.0_hls_weight.py 导出hls的权重和图片
            -util.py 纯python实现读取mnist数据集，用于在pynq上读取数据集
        -output_conv 输出结果存放 
            -py_output python部分的训练结果和量化输出的npy文件；

    -dataset 官方mnist数据集，如果文件夹下没有，就运行1_main.py自动下载

环境：

    conda create -n torch python=3.11
    
    pip3 install torch torchvision torchaudio（或者自己装 GPU 版本，本项目 CPU 版本足够）
    
    pip3 install tqdm
    

网络结构：

    input:      1 * 28 * 28
    
    conv1:      8 * 26 * 26
    
    pooling1:   8 * 13 * 13
    
    conv2:      8 * 11 * 11
    
    pooling2:   8 *  5 *  5
    
    fc1     :   200
