#import general packages
import time
import random
import numpy as np
import argparse
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

#import torch and PYG
import torch
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_model_size, get_data_size, count_parameters


#import my source code
from models import *
from utils import *
from dataloader import GraphDataset

#Define arguments
parser = argparse.ArgumentParser(description='PYG version of Mammography Classification')

# Setting Data path and dataset name
parser.add_argument('--root', type=str, default='/home/linh/Downloads/OCT/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--training_dataset_name', type=str, default='Trainset_Prewitt_v2_224',
                    help='Choose dataset to train')
parser.add_argument('--testing_dataset_name', type=str, default='Testset_Prewitt_v2_224',
                    help='Choose dataset to train')
# Setting hardwares and random seeds
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA to train a model (default: True)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='choose a random seed (default: 42)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='set number of workers (default: 4)')


            
# Learning rate schedule parameters
parser.add_argument('-b','--batch_size', type=int, default=4048, metavar='B',
                    help='input batch size for training (default: 2048')
parser.add_argument('--step_size', type=int, default=20, metavar='SS',
                    help='Set step size for scheduler of learning rate (default: 20')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight_decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

                    
# Setting model configuration
parser.add_argument('--layer_name', type=str, default='GraphConv',
                    help='choose model type either GAT, GCN, or GraphConv (Default: GraphConv')
parser.add_argument('--c_hidden', type=int, default=64,
                    help='Choose numbers of output channels (default: 64')
parser.add_argument('--num_layers', type=int, default=3,
                    help='Choose numbers of Graph layers for the model (default: 3')
parser.add_argument('--dp_rate_linear', type=float, default=0.5,
                    help='Set dropout rate at the linear layer (default: 0.5)')
parser.add_argument('--dp_rate', type=float, default=0.5,
                    help='Set dropout rate at every graph layer (default: 0.5)')

args = parser.parse_args()


if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probaly run with --cuda")
    
    else:
        device_id = torch.cuda.current_device()
        print("***** USE DEVICE *****", device_id, torch.cuda.get_device_name(device_id))
device = torch.device("cuda" if args.cuda else "cpu")
print("==== DEVICE ====", device)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

#############################
training_dataset = GraphDataset(root=args.root, name=args.training_dataset_name, use_node_attr=True)
testing_dataset = GraphDataset(root=args.root, name=args.testing_dataset_name, use_node_attr=True)

training_data_size = len(training_dataset)
#checking some of the data attributes comment out these lines if not needed to check
print('############## INFORMATION OF TRAINING DATA ##################')
print(f'Training Dataset name: {training_dataset}:')
print('==================')
print(f'Number of graphs in training dataset: {len(training_dataset)}')
print(f'Number of features in training dataset: {training_dataset.num_features}')
print(f'Number of classes in training dataset: {training_dataset.num_classes}')

training_data = training_dataset[0]  # Get the first graph object.
print()
print(training_data)
print('==================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes in training dataset: {training_data.num_nodes}')
print(f'Number of edges in training dataset: {training_data.num_edges}')
print(f'Average node degree in training dataset: {training_data.num_edges / training_data.num_nodes:.2f}')
print(f'Has isolated nodes in training dataset: {training_data.has_isolated_nodes()}')
print(f'Has self-loops in training dataset: {training_data.has_self_loops()}')
print(f'Is undirected: {training_data.is_undirected()}')

# Information of Model setting
print("*"*12)
#print(f'number of hidden dim: {args.hidden_dim}')
#print(f'Dropout parameter setting: {args.dropout}')
print("*"*12)


#checking some of the data attributes comment out these lines if not needed to check
print('############## INFORMATION OF TESTING DATA ##################')
print(f'Testing Dataset name: {testing_dataset}:')
print('==================')
print(f'Number of graphs in testing dataset: {len(testing_dataset)}')
print(f'Number of features in testing dataset: {testing_dataset.num_features}')
print(f'Number of classes in testing dataset: {testing_dataset.num_classes}')

testing_data = testing_dataset[0]  # Get the first graph object.
print()
print(testing_data)
print('==================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes in testing dataset: {testing_data.num_nodes}')
print(f'Number of edges in testing dataset: {testing_data.num_edges}')
print(f'Average node degree in testing dataset: {testing_data.num_edges / testing_data.num_nodes:.2f}')
print(f'Has isolated nodes in testing dataset: {training_data.has_isolated_nodes()}')
print(f'Has self-loops in testing dataset: {testing_data.has_self_loops()}')
print(f'Is undirected: {testing_data.is_undirected()}')

# Information of Model setting
print("*"*12)
#print(f'number of hidden dim: {args.hidden_dim}')
#print(f'Dropout parameter setting: {args.dropout}')
print("*"*12)


training_dataset = training_dataset.shuffle()
#this is equivalent of doing
#perm = torch.randperm(len(dataset))
#dataset = dataset[perm]
train_dataset = training_dataset[75820:]
val_dataset   = training_dataset[:75820]
test_dataset  = testing_dataset


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of val graphs: {len(val_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print("**************************")

model = GraphGNNModel(c_in=training_dataset.num_node_features, 
                      c_out=training_dataset.num_classes,
                      layer_name=args.layer_name, 
                      c_hidden=args.c_hidden, 
                      num_layers=args.num_layers, 
                      dp_rate_linear=args.dp_rate_linear, 
                      dp_rate=args.dp_rate).to(device)
print('*****Model size is: ', get_model_size(model))
print("=====Model parameters are: ", count_parameters(model))
print(model)
print("*****Data sizes are: ", get_data_size(data))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
#@profileit()
def train():
    model.train()
    correct = 0
    running_loss = 0
    for data in tqdm(train_loader, desc=(f'Training epoch: {epoch:04d}')):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients. 


#@timeit()
@torch.no_grad()
def test(val_loader):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    running_loss = 0
    for data in tqdm(val_loader, desc=(f'Validation epoch: {epoch:04d}')):  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch) 
        loss = criterion(out, data.y) 
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
        running_loss += loss.item() #* data.num_graphs
    val_loss = running_loss/len(val_loader)
    report = classification_report(y_true, y_pred, digits=4)
    print(report)   
    return correct / len(val_loader.dataset), val_loss # Derive ratio of correct predictions.


start = time.time()
best_val_acc = 0.9
train_accs = []
val_accs = []
train_losses = []
val_losses = []
for epoch in range(1, args.epochs):
    train()
    train_acc, train_loss = test(train_loader)
    val_acc, val_loss = test(val_loader)
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_weight_path = osp.join(args.root + "weights/Graph_" + args.layer_name + "_" + args.training_dataset_name + "_best" + ".pth")
        print('New best model saved to:', save_weight_path)
        torch.save(model.state_dict(), save_weight_path)

    if epoch % 10 == 0:
        print(f'Epoch numbers: {epoch:03d}, Train Acc: {train_acc:.4f},  Validation Acc: {val_acc:.4f}')

    
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    
# Visualization at the end of training
fig, ax = plt.subplots()
ax.plot(train_accs, c="steelblue", label="Training")
ax.plot(val_accs, c="orangered", label="Validation")
ax.plot(train_losses, c="black", label="Training Loss")
ax.plot(val_losses, c="green", label="Validation Loss")
ax.plot()
ax.grid()
ax.legend()
ax.set_xlabel('Epoch Numbers')
ax.set_ylabel('Accuracy and Loss values')
ax.legend(loc='best')
ax.set_title("Accuracy evolution")
#plt.show()
plt.savefig(args.root + "results/Evolution_training_" + args.layer_name + "_" + args.training_dataset_name + ".png")
        
end = time.time()
time_to_train = (end - start)/60
print("Total training time to train on GPU (min):", time_to_train)
print("****End training process here******")

@torch.no_grad()
def inference(loader):
    model.eval()
    correct = 0
    y_pred = []
    y_true = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #correct += int((pred == data.y).sum())  # Check against ground-truth labels

        y_true.extend(data.y.cpu().numpy())
        y_pred.extend(np.squeeze(pred.cpu().numpy().T))
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    cm = confusion_matrix(y_true, y_pred)
    # plot the confusion matrix
    display_labels = ['APC', 'LBB', 'NOR', 'PAB', 'PVC', 'RBB', 'VEB', 'VFW']
    plot_cm(cm=cm, display_labels=display_labels)
    #return torch.sum(y_pred == y_true).item() / len(y_true)      
    return correct / len(loader.dataset) # Derive ratio of correct predictions.

# Inference test set
print("******Start inference on test set*****")
start_2 = time.time()
inference(test_loader)
end_2 = time.time()
time_to_train_2 = (end_2 - start_2)/60
print("Total Inference time to train on GPU (min):", time_to_train_2)

