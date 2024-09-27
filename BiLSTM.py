from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pickle
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn import Softmax as softmax
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, TensorDataset
# from keras.utils.np_utils import to_categorical
# from tflearn.data_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import deque
# from tensorboardX import SummaryWriter
log_dir = './logs'  # Choose your desired log directory
#%%

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def load_labels(label_file):
    labels = {}
    count = 0
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            label = l.strip()
            labels[label] = count
            count += 1
    return labels
#%%
# Replace this with your actual label file path
label_file = r'D:\FFC-mainfromcdrive\retrained_labels.txt'  # changed C to G
labels = load_labels(label_file)
print(type(labels), labels)
#%%
# # Load your data from the .pkl file
# input_data_dump = r'D:\FFC-main\predicted-frames-non-pool-more-ffcresnetnew5epoch-traindata.pkl'
import pickle, torch
import numpy as np
import torch.nn.functional as F
from torch.nn import Softmax as softmax
# input_data_dump = r'D:\FFC-mainfromcdrive\checkpoint\ffc_resnet18\bestnew.pth'
input_data_dump = r'D:\FFC-mainfromcdrive\checkpoint\resnet18\best.pth'
data = torch.load(input_data_dump, map_location=torch.device('cpu'))
#%%
def print_data(data, indent=0):
    spaces = ' ' * indent
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spaces}Key: {key}")
            print_data(value, indent + 2)
    elif isinstance(data, list):
        for i, value in enumerate(data):
            print(f"{spaces}List item {i}:")
            print_data(value, indent + 2)
    elif isinstance(data, torch.Tensor):
        print(f"{spaces}Tensor shape: {data.shape}")
        print(data)
    else:
        print(f"{spaces}{data}")

# print_data(data)
print("Keys in the checkpoint file:", data.keys())
#%%
import model_zoo

# Choose a model from your list
chosen_model_name = "ffc_resnet18"  # For example
my_models = sorted(name for name in model_zoo.__dict__
                   if name.islower() and not name.startswith("__")
                   and callable(model_zoo.__dict__[name]))
print("my models:",my_models, chosen_model_name  )
#%%
# Check if the chosen model name is in the list
if chosen_model_name in my_models:
    # Get the model class from model_zoo
    chosen_model_class = getattr(model_zoo, chosen_model_name)
    
    # Instantiate the chosen model
    ffcresnet = chosen_model_class()
    
    # Load the state dictionary if available
    ffcresnet.load_state_dict(data['state_dict'])
else:
    print("Chosen model not found in your models list.")


#%%
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import torch.nn as nn
print(ffcresnet)
modules = list(ffcresnet.children())[:-1]
ffc_resnet_modified = nn.Sequential(*modules)

# Freeze the parameters of the FFCResNet model
for param in ffc_resnet_modified.parameters():
    param.requires_grad = False

# Check the modified FFCResNet model structure
# print(ffc_resnet_modified)

ffc_resnet = ffc_resnet_modified
# print(ffc_resnet)
#%%
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device) 
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters

input_size = 1  # should match the feature dimension from ResNet output
hidden_size = 64
num_layers = 2
num_classes = 76  # example for classification task

lstm_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
# print(lstm_model)

#%%
num_epochs = 3
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

traindir = r'D:\sandhyajella2\imagedata\train'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512)


# Example list of labels
labels = ["0","1", "2", "3", "4", "5", "6", "7", "8", "9", "a",
    "accident", "all the best", "allergies", "asthma", "b",
    "blood pressure", "breathe", "bye", "c", "cancer", "d",
    "diabetes", "doctor", "e", "ecg", "emergency", "excuse me",
    "f", "fever", "g", "good afternoon", "good evening", "good morning",
    "good night", "h", "headache", "health insurance", "heart attack",
    "hello", "hospital", "how are u", "i", "i am fine", "j", "k", "l",
    "m", "medicine", "my name is", "n", "nice to meet you", "no", "o",
    "operation", "p", "please", "q", "r", "s", "sorry", "stomachache",
    "t", "thank you", "thermometer", "u", "v", "virus", "vomit", "w",
    "welcome", "what is your name", "x", "y", "yes", "z"]  # your complete list of labels

# Initialize label encoder and encode labels
label_encoder = LabelEncoder()
label_encoder.fit(labels)
# label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
# print("Label encoding mapping:", label_mapping)
#print(label_encoder)
for epoch in range(num_epochs):

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for i, (input, target) in enumerate(train_loader):
        with torch.no_grad():
            # Extract features using ResNet
            output = ffc_resnet_modified(input)
            features = output[1]  # Assuming output[1] is the features tensor
        
        # Print the shape of features
        #print(f'Original shape of features: {features.shape}')
        
        # Determine feature dimensions
        batch_size = features.size(0)
        feature_dim = features.size(1)
        
        # Reshape features to (batch_size, seq_len, feature_dim)
        seq_len = 1  # Example sequence length, adjust as necessary
        
        try:
            lstm_input = features.view(batch_size, seq_len, feature_dim)
            #print(f'Reshaped features to: {lstm_input.shape}')
        except RuntimeError as e:
            #print(f'Error reshaping features: {e}')
            continue
        
        # Forward pass through LSTM
        lstm_output = lstm_model(lstm_input)
        #print(lstm_output.shape, target_tensor.shape)
        # Check for shape consistency
        if lstm_output.size(0) != target.size(0):
                print(f"Shape mismatch: LSTM output shape {lstm_output.shape} vs target shape {target.shape}")
                continue

            # Compute loss
        loss = criterion(lstm_output, target)
        # print("loss:",loss)
        total_loss += loss.item()

            # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

            # Calculate accuracy
        _, predicted = torch.max(lstm_output, 1)
        correct_predictions += (predicted == target).sum().item()
        total_predictions += target.size(0)

        print(f"Predicted: {predicted}")
        print(f"True: {target}")

        if (i+1) % 10 == 0:
                accuracy = correct_predictions / total_predictions
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    

    # Print average loss and accuracy for the epoch
    average_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
#%%
# traindir = r'D:\FFC-mainfromcdrive\imagenet_data\train'
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])

# train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize,
#         ]))


# train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=32)

# # model.eval()

# with torch.no_grad():
#         end = time.time()
#         for i, (input, target) in enumerate(train_loader):
#             # if args.gpu is not None:
#             #     input = input.cuda(args.gpu, non_blocking=True)
#             # target = target.cuda(args.gpu, non_blocking=True)
#             # print(type(input), input.size(), target.shape)
#             # compute output
#             output = ffc_resnet_modified(input)
#             # print(type(output), len(output), output[0].shape, output[1].shape) #, output)
#             print(output[1].size(1))
# # label_encoder = LabelEncoder()
# label_encoder.fit(list(labels.keys()))

# # Define the number of frames per video
# num_frames_per_video = 201 # Define the number of frames
# #%%
# # Process the data
# X = []
# y = []
# temp_list = deque()

# for i, frame in enumerate(frames):
#     features = frame[0]
#     actual = frame[1].lower()
#     actual_numeric = label_encoder.transform([actual])[0]
    
#     if len(temp_list) == num_frames_per_video - 1:
#         temp_list.append(features)
#         flat = list(temp_list)
#         X.append(np.array(flat))
#         y.append(actual_numeric)
#         temp_list.clear()
#     else:
#         temp_list.append(features)
#         continue
# print(len(X))
# print(X[1].shape, y[1].shape)
# print(X[15], y[15])
# #%%
# import numpy as np
# X = np.array(X)
# print(type(X))
# y = np.array(y)
# # print(len(labels))
# y = to_categorical(y, len(labels))
# print(y)
# #%%
# print("Dataset shape: ", X.shape)
# # Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print('X_train_shape',(X_train.shape))
# print('Y_train_shape',(y_train.shape))
# # Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# print('X_train_shape',X_train_tensor.shape)
# #y_train_tensor = to_categorical(torch.tensor(y_train, dtype=torch.float32))
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32) 
# print('y_train_tensor',y_train_tensor.shape) 
#  # Using int64 for classification labels
# # y_train_tensor = y_train_tensor.unsqueeze(1)  # Add a new dimension
# # y_train_tensor = to_categorical(y_train_tensor) 
# # print(y_train_tensor.shape)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# print(X_train_tensor.shape)
# #print(y_train_tensor.shape)
# #%%
# # Create PyTorch Dataset and DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Define your LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super(LSTMModel, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size

#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
        
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out

# # Initialize the model

# # X_train = torch.from_numpy(X_train)
# # X_train = X_train.permute(2,1,0)
# input_size = X_train.shape[2]
#  # Modify this based on your feature size
# hidden_size = 6
# num_layers = 1
# output_size = len(labels)  # Update with the number of classes

# model = LSTMModel(input_size, hidden_size, num_layers, output_size)
# print(model)
#%%
# # # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# # Training loop
# num_epochs = 1000
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0.0
#     correct_train = 0
#     total_samples_train = 0
    
#     for batch_X, batch_y in train_loader:
#         optimizer.zero_grad()
#         outputs = model(batch_X)
#         _, predicted = torch.max(outputs.data, 1)
        
#         # Calculate training accuracy for the current batch
#         correct_train += (predicted == torch.argmax(batch_y, dim=1)).sum().item()
#         total_samples_train += batch_y.size(0)
        
#         loss = criterion(outputs, torch.argmax(batch_y, dim=1))
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
        
#     average_loss = total_loss / len(train_loader)
#     print(average_loss)
#     # Check if total_samples_train is non-zero before calculating accuracy
#     if total_samples_train > 0:
#         train_accuracy = (correct_train / total_samples_train) * 100
#         print(correct_train)
#         print(total_samples_train)
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
#     else:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: N/A (Zero samples)')
    
#     # Save the model after each epoch
#     torch.save(model.state_dict(), 'trained_lstm_model.pth')

# Training loop
'''num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_samples_train = 0
    for batch_X, batch_y in train_loader:
        
        optimizer.zero_grad()
        
        outputs = model(batch_X)
        
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
    # Check if total_samples_train is non-zero before calculating accuracy
    if total_samples_train > 0:
        train_accuracy = (correct_train / total_samples_train) * 100
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    else:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: N/A (Zero samples)')
    
   
    #train_accuracy = (correct_train / total_samples_train) * 100
    
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    
    # Save the model after each epoch
    torch.save(model.state_dict(), 'trained_lstm_model.pth')
    
    
   # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    #accuracy = (correct / total_samples) * 100
    
    #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    #torch.save(model.state_dict(), 'trained_lstm_model.pth')
    model.eval()
'''

# %%
