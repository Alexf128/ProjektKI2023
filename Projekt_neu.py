
"""IMPORT LIBRARIES"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

print("Libraries imported - ready to use PyTorch", torch.__version__)


"""LOAD DATASET"""

Train=np.load('W:/Personlich/Weiterbildung/An Alessandro/224/Train_224_imgs.npy')
Train=Train.astype('uint8')

# Show some examples of input data
cols = 3
rows = 3
fig, axs = plt.subplots(rows, cols, figsize=(12, 9))
a=[1, 200, 400, 600, 800, 1000, 1200, 1400, 1600]
i=0
for x in range(rows):
    for y in range(cols):              
        #axs[x, y].set_title(TrainLabels(rnd_idx))
        axs[x, y].imshow(Train[a[i],:,:,:])
        axs[x, y].set_axis_off()
        i=i+1


Train=Train.transpose(0,3,1,2)
TrainLabels=np.load('W:/Personlich/Weiterbildung/An Alessandro/224/Train_224_Label.npy')
TrainLabels=TrainLabels.astype('uint8')
TrainTensor = torch.from_numpy(Train).type(torch.FloatTensor) # transform to torch tensor
TrainLabelsTensor = torch.from_numpy(TrainLabels).type(torch.LongTensor)
my_dataset = TensorDataset(TrainTensor,TrainLabelsTensor ) # create datset
train_loader = DataLoader(my_dataset, batch_size=50,num_workers=0,shuffle=False) # create dataloader

Test=np.load('W:/Personlich/Weiterbildung/An Alessandro/224/Test_224_imgs.npy')
Test=Test.astype('uint8')
Test=Test.transpose(0,3,1,2)
TestLabels=np.load('W:/Personlich/Weiterbildung/An Alessandro/224/Test_224_Label.npy')
TestLabels=TestLabels.astype('uint8')
TestTensor = torch.from_numpy(Test).type(torch.FloatTensor) # transform to torch tensor
TestLabelsTensor = torch.from_numpy(TestLabels).type(torch.LongTensor)
my_dataset_test = TensorDataset(TestTensor,TestLabelsTensor ) # create datset
test_loader = DataLoader(my_dataset_test, batch_size=50,num_workers=0,shuffle=False) # create dataloader



"""Create NN"""
class Net(nn.Module):
    
    # Defining the Constructor
    def __init__(self, num_classes=16):
        super(Net, self).__init__()       
       
        # RGB --> input channels = 3
        # Conv 1 --> 32 Filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Conv 2: 32 --> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
         # Conv 3: 64 --> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Conv 4: 128 --> 256 channels
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # Conv 5: 256 --> 512 channels
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)        
       
        # Max pooling with a kernel size of 2
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Drop layer deletes 20% of the features to help prevent overfitting
        self.drop = nn.Dropout2d(p=0.2)
        
        # 224x224 image tensors will be pooled three times with a kernel size of 2. 224/2/2/2/2/2 is 7.
        # This means that our feature tensors are now 28 x 28, and we've generated 24 of them
        
        # Fully-connected layer
        self.fc = nn.Linear(in_features=7* 7 * 512, out_features=num_classes)

    def forward(self, x):

        # ReLU activation function after layer 1 (convolution 1 and pool)
        x = F.relu(self.pool(self.conv1(x))) 
        
        # Use a ReLU activation function after layer 2
        x = F.relu(self.pool(self.conv2(x)))   
        
        # Use a ReLU activation function after layer 3
        x = F.relu(self.pool(self.conv3(x)))     
        
        # Use a ReLU activation function after layer 4
        x = F.relu(self.pool(self.conv4(x)))  
        
        # Use a ReLU activation function after layer 5
        x = F.relu(self.pool(self.conv5(x)))  
        
        # Select some features to drop to prevent overfitting (only drop during training)
        x = F.dropout(self.drop(x), training=self.training)
        
        # Flatten
        x = x.view(-1, 7 * 7 * 512)
        # Feed to fully-connected layer to predict class
        x = self.fc(self.fc(self.fc(x)))
        # Return class probabilities via a log_softmax function 
        return torch.log_softmax(x, dim=1)
    
device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use cuda (on a cpu, training will take a considerable length of time!)
    device = "cuda"

# Create an instance of the model class and allocate it to the device
model = Net(num_classes=16).to(device)

print(model)


"""Training function"""
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)
        
        # Reset the optimizer
        optimizer.zero_grad()
        
        # Push the data forward through the model layers
        output = model(data)
        
        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()
        
        # Backpropagate
        loss.backward()
        optimizer.step()
        
        # Print metrics 
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss

"""Testing function ---> NACH KORREKTUR DER AKTUELLEN PROBLEMEN"""
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)
            
            # Get the predicted classes for this batch
            output = model(data)
            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()
            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss

    
    # Use an "Adam" optimizer to adjust weights
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 10 epochs 
epochs = 10000 
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        
        
#Plot
plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()   
    
    
""" Evaluate Model"""
# Defining Labels and Predictions
truelabels = []
predictions = []
model.eval()
print("Getting predictions from test set...")
for data, target in test_loader:
    for label in target.data.numpy():
        truelabels.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction) 


classes=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# Plot the confusion matrix
cm = confusion_matrix(truelabels, predictions)
tick_marks = np.arange(len(classes))

df_cm = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (7,7))
sns.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.xlabel("Predicted Shape", fontsize = 20)
plt.ylabel("True Shape", fontsize = 20)
plt.show() 
    
   
    
    
    