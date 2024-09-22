import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1) # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
      self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1) # We double the feature maps for every conv layer as in pratice it is really good.
      self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
      self.fc1 = nn.Linear(32*32*64, 500) # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
      self.dropout1 = nn.Dropout(0.5)
      self.fc2 = nn.Linear(500, 2) # output nodes are 10 because our dataset have 10 different categories
      
    def forward(self, x):
      x = F.relu(self.conv1(x)) #Apply relu to each output of conv layer.
      x = F.max_pool2d(x, 2, 2) # Max pooling layer with kernal of 2 and stride of 2
      x = F.relu(self.conv2(x))
      x = F.max_pool2d(x, 2, 2)
      x = F.relu(self.conv3(x))
      x = F.max_pool2d(x, 2, 2)
      x = x.view(-1, 32*32*64) # flatten our images to 1D to input it to the fully connected layers
      x = F.relu(self.fc1(x))
      x = self.dropout1(x) # Applying dropout b/t layers which exchange highest parameters. This is a good practice
      x = self.fc2(x)
      
      return x
    
PATH = './cifar_net.pth' # path where the model will be saved to

def trainmodel(net, trainloader, start_from_scratch):
    if not start_from_scratch:
        net.load_state_dict(torch.load(PATH, weights_only=True))
        print('Loaded old model')
    else:
        print('Training a new model')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    # after calling optimizer.step(), optimiser iterates over each parameter it was initialised whit
    # thus it will update the parameters of net (weights of edges and biases)
    # net.grad (torch.nn.Module.grad) attribute stores the gradients after the backward pass
    # optimizer.step receives net.parameters() as input, and updates them after .grad are known
    
    print('Starting training')
    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
                
    torch.save(net.state_dict(), PATH)
    print('Finished Training and Saving Model')
    
def testmodel(net, testloader, classes):
    net.load_state_dict(torch.load(PATH, weights_only=True))

    total = 0
    correct = 0
    
    print('Starting testing')
    for data in testloader:
        images, labels = data
        
        output = net(images)
        scores, predicted = torch.max(output, 1)
        
        ground_truth = []
        for j in range (len(labels)):
            ground_truth.append(classes[labels[j]])
            
        predicted_labels = []
        for j in range (len(predicted)):
            predicted_labels.append(classes[predicted[j]])
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print(ground_truth, predicted_labels)
        #print('Finished Testing Model')
        
    print(f'Finished testing, accuracy is {correct/total*100} %')
    
def loader(path):
    return Image.open(path)

def get_coins(net, classes, transform):
    img = cv2.imread('./two_coins.JPG')
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    kernel = np.ones((5,5),np.float32)/25
    img_gray = cv2.filter2D(img_gray,-1,kernel)
    
    ret, img_binary = cv2.threshold(img_gray, 75, 255, 0)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    mx = 0
    
    for i, contour in enumerate(contours, 0): 
        if i == 0:
            continue 
        
        mx = max(mx, cv2.contourArea(contour))
        
    coin_count = 0
    coins = []
    
    for i, contour in enumerate(contours, 0): 
        if i == 0:
            continue 
        
        if cv2.contourArea(contour) >= mx / 1.4:
            # find radius
            (x,y),radius = cv2.minEnclosingCircle(contour) 
            x = int(x)
            y = int(y)
            
            center = (x, y)
            radius = int(radius * 1.5) 
            
            img_coin = img[y-radius:y+radius, x-radius:x+radius]
            coins.append(img_coin)
            cv2.imwrite('./coins_images/coin' + str(coin_count) + ".jpg", img_coin)
            coin_count += 1
                
            #cv2.circle(img_rgb,center,radius,(0,0,255),10);
        
    sum = 0
    
    for idx, img_coin in enumerate(coins, 0):
        bigger = cv2.resize(img_coin, (256, 256))
        tensor = transform(bigger)
        output = net(tensor)
        
        scores, predicted = torch.max(output, 1)
        
        for j in range (len(predicted)):
            sum += ord(classes[predicted[j]][0]) - ord('0')
            print("Prediciton # 1:", classes[predicted[j]])
    
    print(f'The sum of the coins is £{sum}')

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.ImageFolder('./train', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.ImageFolder('./test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('1 pound', '2 pound')

    net = Net()
    
    trainmodel(net, trainloader, False)
    testmodel(net, testloader, classes)
    
    for i in range(5):    
        get_coins(net, classes, transform)
    
if __name__ == "__main__":
    main()