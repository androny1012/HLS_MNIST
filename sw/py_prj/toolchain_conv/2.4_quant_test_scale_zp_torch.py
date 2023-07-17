import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/mnist/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=1, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define a floating point model
c = 8
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,c,kernel_size=3,stride=1,padding=0,bias=False)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(c,c,kernel_size=3,stride=1,padding=0,bias=False)
        self.fc1 = nn.Linear(c * 5 * 5,10)
        self.relu = torch.nn.ReLU()
        self.input_image = None
        self.conv1_out = None
        self.conv2_out = None
        self.fc1_out = None

    def forward(self, x):
        self.input_image = x
        self.conv1_out = self.conv1(x)
        x = self.pool(self.relu(self.conv1_out))
        self.conv2_out = self.conv2(x)
        x = self.pool(self.relu(self.conv2_out))
        x = x.reshape(-1, c * 5 * 5)
        self.fc1_out = self.fc1(x)
        x = self.relu(self.fc1_out)

        return x

    def get_conv_out(self):
        return self.input_image,self.conv1_out,self.conv2_out,self.fc1_out
    
float_model_name = './py_prj/output_conv/py_output/1_model.pth'
quant_model_name = './py_prj/output_conv/py_output/1_model_quant.pth'

def load_quant_model(float_model_name,quant_model_name):
    model = Net()
    network_state_dict = torch.load(float_model_name,map_location=torch.device('cpu'))
    model.load_state_dict(network_state_dict)
    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model=nn.Sequential(quant,model,dequant)

    quant_model = quant_model.to('cpu')
    quant_model.qconfig = torch.quantization.default_qconfig
    quant_model=torch.quantization.prepare(quant_model, inplace=False)
    quant_model=torch.quantization.convert(quant_model, inplace=False)

    state_dict_t = torch.load(quant_model_name)
    quant_model.load_state_dict(state_dict_t)
    quant_model=quant_model.to('cpu')
    return quant_model


model_int8=load_quant_model(float_model_name,quant_model_name)


model_int8.eval()
with torch.no_grad():
    for data, target in test_loader:
        output = model_int8(data.to(device))
        pred = output.data.max(1, keepdim=True)[1]
        # print(pred,target)
        break

# print(data[0][0][7])

id = 0
for i in model_int8.children():
    # print(i)
    if(id == 1):
        break
    id = id + 1
net = i
# print(model_int8)

input_image, conv1_out, conv2_out, fc1_out = net.get_conv_out()
# print(conv1_out.shape)
# print(conv1_out.dtype)

iscale=model_int8.state_dict()['0.scale'].item()
izp=model_int8.state_dict()['0.zero_point'].item()
print(iscale,izp)
# print(input_image[0][0][7])
# print(conv1_out[0][0][5])
input_int8 = np.around(np.asarray([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3307, 0.7244, 0.6220,
        0.5906, 0.2362, 0.1417, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000]) / iscale) + izp

print(input_int8)

conv1_scale=model_int8.state_dict()['1.conv1.scale'].item()
conv1_zp=model_int8.state_dict()['1.conv1.zero_point'].item()
conv1_out_int8 = np.around(np.asarray([0.0000, 0.0000, 0.0000, 0.0000, 0.1383, 0.6917, 1.2450, 1.4525, 1.1758,
        0.7608, 0.2767, 0.0692, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]) / conv1_scale) + conv1_zp

print(conv1_out_int8)

conv2_scale=model_int8.state_dict()['1.conv2.scale'].item()
conv2_zp=model_int8.state_dict()['1.conv2.zero_point'].item()
conv2_out_int8 = np.around(np.asarray([-3.5166, -7.4240, -6.6426, -6.2518, -7.0333, -7.8148, -8.2055, -8.2055,
        -3.9074, -0.3907, -0.3907]) / conv2_scale) + conv2_zp

# print(conv2_out[0][0][2])
print(conv2_out_int8)

fc1_scale=model_int8.state_dict()['1.fc1.scale'].item()
fc1_zp=model_int8.state_dict()['1.fc1.zero_point'].item()

fc1_out_int8 = np.around(np.asarray([ -5.3783,  -9.2200,  -3.0733,  -3.0733, -26.1233, -15.3667, -34.5750,
          16.9033, -10.7567,  -3.8417]) / fc1_scale) + fc1_zp

# print(fc1_out)
print(fc1_out_int8)


# for layer in model_int8.named_modules():
#     print(layer[0],layer[1])
# print(model_int8.named_modules)
# conv1_out,conv2_out=model_int8.get_conv_out()
# print(conv1_out.shape)

print(model_int8)