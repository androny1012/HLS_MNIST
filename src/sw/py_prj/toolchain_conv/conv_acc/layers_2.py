# coding:utf-8
import numpy as np
import time

def img2col(input, height_out, width_out, kernel_size, stride):
    output = np.zeros([input.shape[0], input.shape[1], kernel_size*kernel_size, height_out*width_out])
    height = int((input.shape[2] - kernel_size) / stride + 1)
    width = int((input.shape[3] - kernel_size) / stride + 1)
    for idxh in range(height):
        for idxw in range(width):
            output[:, :, :, idxh * width + idxw] = input[:, :, idxh * stride : idxh * stride + kernel_size, idxw * stride : idxw * stride + kernel_size].reshape(input.shape[0], input.shape[1], -1)
    return output

def col2img(input, height_pad, width_pad, kernel_size, channel, padding, stride):
    output = np.zeros([input.shape[0], channel, height_pad, width_pad])
    input = input.reshape(input.shape[0], channel, -1, input.shape[2])
    height = int((height_pad - kernel_size) / stride + 1)
    width = int((width_pad - kernel_size) / stride + 1)
    for idxh in range(height):
        for idxw in range(width):
            output[:, :, idxh * stride : idxh * stride + kernel_size, idxw * stride : idxw * stride + kernel_size] += input[:, :, :, idxh * width + idxw].reshape(input.shape[0], channel, kernel_size, -1)
    return output[:, :, padding : height_pad - padding, padding : width_pad - padding]

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, channel, height, width])
    image_col = []
    for b in range(image.shape[0]):
        for i in range(0, image.shape[2] - ksize + 1, stride):
            for j in range(0, image.shape[3] - ksize + 1, stride):
                col = image[b, :, i:i + ksize, j:j + ksize].reshape([-1])
                image_col.append(col)
    image_col = np.array(image_col)
    return image_col #[N, ((H-k)/s+1)*((w-k)/s+1), k*k*cin]

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride, type=0):
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        # print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = int((height - self.kernel_size) / self.stride + 1)
        width_out = int((width - self.kernel_size) / self.stride + 1)
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        #(3.3)
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * \
                        self.input_pad[idxn, :, hs:hs+self.kernel_size, ws:ws+self.kernel_size]) + \
                        self.bias[idxc]
        self.forward_time = time.time() - start_time
        return self.output

    def forward_speedup(self, input):
        # TODO: 改进forward函数，使得计算加速
        start_time = time.time()
        self.input = input
        N = self.input.shape[0]
        cin = self.weight.shape[0]
        cout = self.weight.shape[3]
        height = self.input.shape[2] + self.padding * 2
        width = self.input.shape[3] + self.padding * 2
        #1.padding.
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]] = self.input
        height_out = int((height - self.kernel_size) / self.stride + 1)
        width_out = int((width - self.kernel_size) / self.stride + 1)
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        
        #2.weight reshape
        col_weight = np.reshape(self.weight, [-1,cout]) #cin,k,k,cout-> cin*k*k,cout
        #3.col reshape. can be speed up too.
        self.col_image = im2col(self.input_pad, self.kernel_size, self.stride) #N,Cin,H,W -> N,(height_out)*(width_out),cin*k*k
        
        #4.matrix multiply
        self.output = np.dot(self.col_image, col_weight) + self.bias
        #5.reshape to ours.
        self.output = np.reshape(self.output, np.hstack(([N],[height_out],[width_out],[cout]))) #N,hight_out*width_out,Cout -> N,hight_out, width_out, Cout->N,Cout,hight_out,width_out
        self.output = np.transpose(self.output, [0,3,1,2])
        self.forward_time = time.time() - start_time
        return self.output
    def backward_speedup(self, top_diff):
        # TODO: 改进backward函数，使得计算加速
        start_time = time.time()
        N = self.input.shape[0]
        cin = self.weight.shape[0]
        cout = self.weight.shape[3]
        pad_height = top_diff.shape[2] + self.padding * 2
        pad_width = top_diff.shape[3] + self.padding * 2
        
        #1.get d_weight and d_bias
        # bottom_diff = np.zeros(self.input_pad.shape)
        col_diff = np.reshape(top_diff, [cout, -1]).T
        self.d_weight = np.dot(self.col_image.T, col_diff).reshape(self.weight.shape)
        self.d_bias = np.sum(col_diff, axis=0)
        #2.pad top_diff
        pad_diff = np.zeros(shape=(top_diff.shape[0], top_diff.shape[1], pad_height, pad_width))
        pad_diff[:, :, self.padding:self.padding+top_diff.shape[2], self.padding:self.padding+top_diff.shape[3]]=top_diff
        #3.flip weight(xuanzhuan 180)
        #our weight:(cin, k, k, cout) 
        weight_reshape = np.reshape(self.weight, [cin,-1,cout])
        flip_weight = weight_reshape[:,::-1,...]
        flip_weight = flip_weight.swapaxes(0,2)
        col_flip_weight = flip_weight.reshape([-1, cin]) #cout*k*k, cin

        #4.get bottom diff
        col_pad_diff = im2col(pad_diff, self.kernel_size, self.stride)
        bottom_diff = np.dot(col_pad_diff, col_flip_weight)
        #reshape
        # import pdb
        # pdb.set_trace()
        bottom_diff = np.reshape(bottom_diff, [N, self.input.shape[2], self.input.shape[3], self.input.shape[1]])#n*w*w*c -> n,h,w,c
        bottom_diff = np.transpose(bottom_diff, [0, 3, 1, 2]) #n,h,w,c->n,c,h,w
        self.backward_time = time.time() - start_time
        return bottom_diff
    def backward_raw(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        #print 'input_pad.shape', self.input_pad.shape
        #print 'top_diff.shape', top_diff.shape
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失(3.5)
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        self.d_weight[:, :, :, idxc] += np.dot(top_diff[idxn,idxc,idxh,idxw],self.input_pad[idxn,:,hs:hs+self.kernel_size, ws:ws+self.kernel_size])
                        self.d_bias[idxc] += top_diff[idxn,idxc,idxh,idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size] += \
                        top_diff[idxn,idxc,idxh,idxw] * self.weight[:,:,:,idxc]
        #(3.6)!!!bottom_diff.shape[2]
        bottom_diff = bottom_diff[:,:,self.padding:bottom_diff.shape[2]-self.padding,self.padding:bottom_diff.shape[3]-self.padding]
        self.backward_time = time.time() - start_time
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):
        self.weight += - lr * self.d_weight
        self.bias += - lr * self.d_bias
    def load_param(self, weight, bias):
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def get_forward_time(self):
        return self.forward_time
    def get_backward_time(self):
        return self.backward_time

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride, type=0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.forward = self.forward_raw
        self.backward = self.backward_raw_book
        if type == 1:  # type 设为 1 时，使用优化后的 foward 和 backward 函数
            self.forward = self.forward_speedup
            self.backward = self.backward_speedup
        # print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward_raw(self, input):
        start_time = time.time()
        self.input = input # [N, C, H, W]
        self.max_index = np.zeros(self.input.shape)
        height_out = int((self.input.shape[2] - self.kernel_size) / self.stride + 1)
        # print('height_out: ', height_out, 'input: ', self.input.shape[2])
        width_out = int((self.input.shape[3] - self.kernel_size) / self.stride + 1)
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        #(3.7)
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, hs:hs+self.kernel_size, ws:ws+self.kernel_size])
                        curren_max_index = np.argmax(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        curren_max_index = np.unravel_index(curren_max_index, [self.kernel_size, self.kernel_size])
                        self.max_index[idxn, idxc, idxh*self.stride+curren_max_index[0], idxw*self.stride+curren_max_index[1]] = 1
        return self.output
    def forward_speedup(self, input):
        # 改进forward函数，使得计算加速
        start_time = time.time()
        
        self.input = input # [N, C, H, W]
        self.height_out = int((self.input.shape[2] - self.kernel_size) / self.stride + 1)
        self.width_out = int((self.input.shape[3] - self.kernel_size) / self.stride + 1)

        self.input_col = img2col(self.input, self.height_out, self.width_out, self.kernel_size, self.stride).reshape(self.input.shape[0], self.input.shape[1], -1, self.height_out, self.width_out)
        output = self.input_col.max(axis=2, keepdims=True)
        self.max_index = (self.input_col == output)
        self.output = output.reshape(self.input.shape[0], self.input.shape[1], self.height_out, self.width_out)

        return self.output
    def backward_speedup(self, top_diff):
        # 改进backward函数，使得计算加速

        pool_diff = (self.max_index * top_diff[:, :, np.newaxis, :, :]).reshape(self.input.shape[0], -1, self.height_out * self.width_out)
        bottom_diff = col2img(pool_diff, self.input.shape[2], self.input.shape[3], self.kernel_size, self.input.shape[1], 0, self.stride)

        return bottom_diff
    def backward_raw_book(self, top_diff):
        # print(top_diff.shape, self.input.shape)
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失#(3.8)
                        hs = idxh * self.stride
                        ws = idxw * self.stride
                        max_index = np.argmax(self.input[idxn,idxc,hs:hs+self.kernel_size,ws:ws+self.kernel_size])
                        max_index = np.unravel_index(max_index, (self.kernel_size, self.kernel_size))
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        # print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        self.input = np.transpose(input, [0, 2, 3, 1])
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        return self.output
    def backward(self, top_diff):
        assert list(top_diff.shape[1:]) == list(self.output_shape)
        top_diff = np.transpose(top_diff, [0, 3, 1, 2])
        bottom_diff = top_diff.reshape([top_diff.shape[0]] + list(self.input_shape))
        return bottom_diff