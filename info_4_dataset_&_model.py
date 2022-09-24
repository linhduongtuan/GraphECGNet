#import general packages
import time
import random
import numpy as np
import argparse
import os.path as osp
import matplotlib.pyplot as plt

#import torch and PYG
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.profile import get_model_size, get_data_size, count_parameters

#import my source code
from models import *
from utils import *
from dataloader import GraphDataset

#Define arguments
parser = argparse.ArgumentParser(description='PYG version of Mammography Classification')

# Setting Data path and dataset name
parser.add_argument('--root', type=str, default='/home/linh/Downloads/data/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset_name', type=str, default='BIRAD_Prewitt_v2',
                    help='Choose dataset to train')
# Setting hardwares and random seeds
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA to train a model')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='choose a random seed (default: 42)')

# Setting training parameters
parser.add_argument('-b','--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 512')

# Setting model configuration
parser.add_argument('--layer_name', type=str, default='GraphConv',
                    help='choose model type either GATConv, GCNConv, or GraphConv (Default: GraphConv')
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
dataset = GraphDataset(root=args.root, name=args.dataset_name, use_node_attr=True)
data_size = len(dataset)
#checking some of the data attributes comment out these lines if not needed to check
print()
print(f'Dataset name: {dataset}:')
print('==================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.
print()
print(data)
print('==================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

# Information of Model setting
print("*"*12)
#print(f'number of hidden dim: {args.hidden_dim}')
#print(f'Dropout parameter setting: {args.dropout}')
print("*"*12)


dataset = dataset.shuffle()
#this is equivalent of doing
#perm = torch.randperm(len(dataset))
#dataset = dataset[perm]

train_dataset = dataset[:6700]
val_dataset = dataset[6700:8150]
test_dataset = dataset[8150:]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

model = GraphGNNModel(c_in=dataset.num_node_features, 
                      c_out=dataset.num_classes,
                      layer_name=args.layer_name, 
                      c_hidden=args.c_hidden, 
                      num_layers=args.num_layers, 
                      dp_rate_linear=args.dp_rate_linear, 
                      dp_rate=args.dp_rate).to(device)
print('*****Model size is: ', get_model_size(model))
print("=====Model parameters are: ", count_parameters(model))
print(model)
print("*****Data sizes are: ", get_data_size(data))
