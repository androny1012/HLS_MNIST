import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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

        # return F.log_softmax(x, dim=1)
        return x  # for test

def test_model_fp32():
    model_fp32.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model_fp32(data.to(device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()

    print('\nmodel_fp32 Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def test_model_int8():
    model_int8.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model_int8(data.to(device))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()

    print('\nmodel_int8 Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == "__main__":
    
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/mnist/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=1000, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    c = 8
    model_path = './py_prj/output_conv/py_output/1_model.pth'
    model_fp32 = Net()
    network_state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    model_fp32.load_state_dict(network_state_dict)
    model_fp32.eval()

    test_model_fp32()


    quant = torch.quantization.QuantStub()
    dequant = torch.quantization.DeQuantStub()
    quant_model=nn.Sequential(quant,model_fp32,dequant)
    quant_model = quant_model.to('cpu')
    quant_model.qconfig = torch.quantization.default_qconfig
    # quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quant_model=torch.quantization.prepare(quant_model, inplace=True)

    with torch.no_grad():
        for data, target in test_loader:
            output = quant_model(data.to(device))

    quant_model_name = './py_prj/output_conv/py_output/1_model_quant.pth'
    model_int8=torch.quantization.convert(quant_model, inplace=True)
    torch.save(model_int8.state_dict(),quant_model_name)

    test_model_int8()