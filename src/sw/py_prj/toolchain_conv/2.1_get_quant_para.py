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
    batch_size=1000, shuffle=True)

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


    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.reshape(-1, c * 5 * 5)
        x = self.relu(self.fc1(x))

        return F.log_softmax(x, dim=1)

quant_model_name = './py_prj/output_conv/py_output/1_model_quant.pth'

model_fp32 = Net()
model_fp32.eval()

quant = torch.quantization.QuantStub()
dequant = torch.quantization.DeQuantStub()
quant_model=nn.Sequential(quant,model_fp32,dequant)
quant_model = quant_model.to('cpu')
quant_model.qconfig = torch.quantization.default_qconfig
quant_model=torch.quantization.prepare(quant_model, inplace=True)
model_int8=torch.quantization.convert(quant_model, inplace=True)
state_dict_t = torch.load(quant_model_name)
quant_model.load_state_dict(state_dict_t)

conv1_weight = model_int8.state_dict()['1.conv1.weight'].int_repr().numpy().astype(np.int8)
conv2_weight = model_int8.state_dict()['1.conv2.weight'].int_repr().numpy().astype(np.int8)
fc1_weight = model_int8.state_dict()['1.fc1._packed_params._packed_params'][0].int_repr().numpy().astype(np.int8)

np.save("./py_prj/output_conv/py_output/2_conv1_weight_q.npy",conv1_weight)
np.save("./py_prj/output_conv/py_output/2_conv2_weight_q.npy",conv2_weight)
np.save("./py_prj/output_conv/py_output/2_fc1_weight_q.npy",fc1_weight)

def generate_para_list(quant_model,type,index):
    cscale=quant_model.state_dict()['1.'+type+str(index)+'.scale'].item()
    czp=quant_model.state_dict()['1.'+type+str(index)+'.zero_point'].item()
    if(type == 'fc'):
        wscale=torch.q_scale(quant_model.state_dict()['1.'+type+str(index)+'._packed_params._packed_params'][0])
        w=quant_model.state_dict()['1.'+type+str(index)+'._packed_params._packed_params'][0].int_repr().numpy().astype(np.int8)
        b=0
    elif(type == 'conv'):
        wscale=torch.q_scale(quant_model.state_dict()['1.'+type+str(index)+'.weight'])
        w=quant_model.state_dict()['1.'+type+str(index)+'.weight'].int_repr().numpy().astype(np.int8)        
        b=quant_model.state_dict()['1.'+type+str(index)+'.bias']
    return w,b,wscale,cscale,czp

def generate_quant_para(iscale,izp,wscale,cscale,czp):
    import math
    #缩写的含义分别为input scale,input zero_point,weight scale,conv scale,conv zero_point
    oscale=cscale
    ozp=czp
    bscale = iscale * wscale
    m=iscale*wscale/oscale
    base,expr=math.frexp(m)
    mult=round(base*(2**15))
    shift=-expr
    return oscale,ozp,bscale,mult,shift


iscale=model_int8.state_dict()['0.scale'].item()
izp=model_int8.state_dict()['0.zero_point'].item()
print(iscale,izp)

w,b,wscale,cscale,czp=generate_para_list(model_int8,'conv',1)
oscale,ozp,bscale,mult,shift=generate_quant_para(iscale,izp,wscale,cscale,czp)
print(oscale,ozp,bscale,mult,shift)

iscale=oscale
izp=ozp
w,b,wscale,cscale,czp=generate_para_list(model_int8,'conv',2)
oscale,ozp,bscale,mult,shift=generate_quant_para(iscale,izp,wscale,cscale,czp)
print(oscale,ozp,bscale,mult,shift)


iscale=oscale
izp=ozp
w,b,wscale,cscale,czp=generate_para_list(model_int8,'fc',1)
oscale,ozp,bscale,mult,shift=generate_quant_para(iscale,izp,wscale,cscale,czp)
print(oscale,ozp,bscale,mult,shift)







