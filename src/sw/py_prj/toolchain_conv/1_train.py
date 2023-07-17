import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


model_path = './py_prj/output_conv/py_output/1_model.pth'
opt_path = './py_prj/output_conv/py_output/1_optimizer.pth'
n_epochs = 1
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 100
random_seed = 1
torch.manual_seed(random_seed)
train_or_not = False
load_or_not = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/mnist/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                #    torchvision.transforms.Normalize(
                                #        (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./dataset/mnist/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                               ])),
    batch_size=batch_size_test, shuffle=True)
 
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

c = 8

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
       
        self.conv1 = nn.Conv2d(1,c,kernel_size=3,stride=1,padding=0,bias=False)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(c,c,kernel_size=3,stride=1,padding=0,bias=False)
        self.fc1 = nn.Linear(c * 5 * 5,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.reshape(-1, c * 5 * 5)
        x = F.relu(self.fc1(x))
        return F.log_softmax(x, dim=1)
 
network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
 
if load_or_not:
    network_state_dict = torch.load(model_path,map_location=torch.device('cpu'))
    network.load_state_dict(network_state_dict)
    optimizer_state_dict = torch.load(opt_path)
    optimizer.load_state_dict(optimizer_state_dict)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
 
 
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
            loss = loss.to('cpu')
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
 
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.to(device).data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if train_or_not:
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
test()

