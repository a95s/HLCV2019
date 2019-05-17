import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt

import copy


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
input_size = 3
num_classes = 10
#hidden_size =   [32,  64,  64,  64,  64,  64]
hidden_size = [128, 512, 512, 512, 512, 512]
num_epochs = 20
batch_size = 4
learning_rate = 2e-3
learning_rate_decay = 0.95
reg=0.001
num_training= 49000
num_validation =1000
norm_layer = None

#print(hidden_size)

model_dir = 'model_dir'

#-------------------------------------------------
# Load the CIFAR-10 dataset
#-------------------------------------------------
#################################################################################
# TODO: Q3.a Chose the right data augmentation transforms with the right        #
# hyper-parameters and put them in the data_aug_transforms variable             #
#################################################################################
data_aug_transforms = []

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

valset = torch.utils.data.Subset(trainset, list(range(48000, 48999)))

valloader = torch.utils.data.DataLoader(dataset=valset,
                                        batch_size=4,
                                        shuffle=False)
"""
# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

norm_transform = transforms.Compose(data_aug_transforms+[transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                           train=True,
                                           transform=norm_transform,
                                           download=False)

test_dataset = torchvision.datasets.CIFAR10(root='datasets/',
                                          train=False,
                                          transform=test_transform
                                          )

#-------------------------------------------------
# Prepare the training and validation splits
#-------------------------------------------------
mask = list(range(num_training))
train_dataset = torch.utils.data.Subset(cifar_dataset, mask)
#train_dataset = torch.utils.data.Subset(trainset, mask)
mask = list(range(num_training, num_training + num_validation))
val_dataset = torch.utils.data.Subset(cifar_dataset, mask)
#val_dataset = torch.utils.data.Subset(testset, mask)

#-------------------------------------------------
# Data loader
#-------------------------------------------------
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
test_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


#-------------------------------------------------
# Convolutional neural network (Q1.a and Q2.a)
# Set norm_layer for different networks whether using batch normalization
#-------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
        #       padding_mode='zeros')

        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

        # Linear(in_features, out_features, bias=True)
        kernel_size = 3

        self.conv1 = nn.Conv2d(input_size, hidden_layers[0], kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_layers[0], hidden_layers[1], kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_layers[1], hidden_layers[2], kernel_size, stride=1, padding=1)
        self.conv4 = nn.Conv2d(hidden_layers[2], hidden_layers[3], kernel_size, stride=1, padding=1)
        self.conv5 = nn.Conv2d(hidden_layers[3], hidden_layers[4], kernel_size, stride=1, padding=1)

        self.fc1 = nn.Linear(hidden_layers[4],hidden_layers[5])
        self.fc2 = nn.Linear(hidden_layers[5], num_classes)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #return out


#-------------------------------------------------
# Calculate the model size (Q1.b)
# if disp is true, print the model parameters, otherwise, only return the number of parameters.
#-------------------------------------------------
def PrintModelSize(model, disp=True):
    #################################################################################
    # TODO: Implement the function to count the number of trainable parameters in   #
    # the input model. This useful to track the capacity of the model you are       #
    # training                                                                      #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model_sz = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total number of trainable parameters: {}'.format(model_sz))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return model_sz

#-------------------------------------------------
# Calculate the model size (Q1.c)
# visualize the convolution filters of the first convolution layer of the input model
#-------------------------------------------------
def VisualizeFilter(model):
    #################################################################################
    # TODO: Implement the functiont to visualize the weights in the first conv layer#
    # in the model. Visualize them as a single image fo stacked filters.            #
    # You can use matlplotlib.imshow to visualize an image in python                #
    #################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    fig=plt.figure(figsize=(3, 3))
    columns = 8  # 16
    rows    = 4  # 8
    theData = model.layers[0].weight.data
    for i in range(1, columns * rows + 1):
        maxVal = torch.max(theData[i - 1])
        minVal = torch.min(theData[i - 1])
        img    = (theData[i -1] - minVal) / (maxVal - minVal)
        fig.add_subplot(rows , columns, i).axis('off')

        if str(device) == "cuda":
            #print("converting to cpu")
            imgcpu = img.cpu()
            plt.imshow(imgcpu)
        else:
            plt.imshow(img)

    plt.show()

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

#======================================================================================
# Q1.a: Implementing convolutional neural net in PyTorch
#======================================================================================
# In this question we will implement a convolutional neural networks using the PyTorch
# library.  Please complete the code for the ConvNet class evaluating the model
#--------------------------------------------------------------------------------------

model = ConvNet(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# Q2.a - Initialize the model with correct batch norm layer

#model.apply(weights_init)
# Print the model
print(model)
# Print model size
#======================================================================================
# Q1.b: Implementing the function to count the number of trainable parameters in the model
#======================================================================================
PrintModelSize(model)
#======================================================================================
# Q1.a: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
#======================================================================================
#VisualizeFilter(model)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)

best_acc = 0.

# Train the model
lr = learning_rate
#total_step = len(train_loader)
#for epoch in range(num_epochs):
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
    #for i, data in enumerate(trainloader,0):
        # Move tensors to the configured device
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        #print(images.size())
        optimizer.zero_grad()
        #model.zero_grad()
        outputs = model(images)
        # print("-------")
        # print(outputs)
        # print("-------")
        # print("-------")
        # print(images)
        # print("-------")
        # print("-------")
        # print(labels)
        # print("-------")
        # print(outputs.size())
        loss = criterion(outputs, labels)

        # Backward and optimize

        loss.backward()
        optimizer.step()
        #if i == 100: exit(0)
        #if (i+1) % 2000 == 1999:
        #    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #          .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
        #for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)

            # print("-------")
            # print(images)
            # print("-------")
            # print("-------")
            # print(labels)
            # print("-------")
            # exit(0)
            outputs = model(images)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Validataion accuracy is: {} %'.format(100 * correct / total))
        #################################################################################
        # TODO: Q2.b Implement the early stopping mechanism to save the model which has #
        # acheieved the best validation accuracy so-far.                                #
        #################################################################################
        best_model = None
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        acc =  100 * correct / total
        if acc >= best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model.train()

print("Finished Training")

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
model.eval()
#################################################################################
# TODO: Q2.b Implement the early stopping mechanism to load the weights from the#
# best model so far and perform testing with this model.                        #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# load best model weights
#model.load_state_dict(best_model_wts)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
    #for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if total == 1000:
            break

    print('Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Q1.c: Implementing the function to visualize the filters in the first conv layers.
# Visualize the filters before training
VisualizeFilter(model)
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')


