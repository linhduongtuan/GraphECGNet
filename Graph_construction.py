import cv2
import os
import glob
import time
import pandas as pd
import numpy as np

#define globals required through out the whole program
edges           = [] #containing all edge tuple
attrs           = [] #countaining list of attribute of all nodes
graph_id        = 1 #id of latest graph
node_id         = 1 #id of latest node
graph_indicator = [] #containing graph-id for each node
node_labels     = [] #containing labels for all node
graph_labels    = []#containing labels for all graph

#activity-label vs activity-name mapping (4-class)
activity_map    = {}
activity_map[1] = 'Epilepsy'
activity_map[2] = 'No_Epilepsy'



#z-score normalization
def normalize(arr):
    arr = np.array(arr)
    m   = np.mean(arr)
    s   = np.std(arr)
    return (arr - m)/s

#generate graph for a given edge-image file
def generate_graphs(filename, node_label, activity_map):
    print(" ... Reading image: " + filename+" ...")
    global node_id, edges, attrs, graph_id, node_labels, graph_indicator
    cnt           = 0
    img           = cv2.imread(filename)
    dim1, dim2, _ = img.shape
    attrs1        = []

    print("Image type: " + activity_map[node_label] + "\nPixel matrix is of: " + str(dim1) + "x" + str(dim2))
    img1 = img.copy()
    nodes = np.full((dim1, dim2), -1)
    edge = 0
    for i in range(dim1):
        for j in range(dim2):
            #considering pixel as node if pixel-value>=128
            b, _, _ = img[i][j]
            if(b >= 128):
                nodes[i][j] = node_id
                attrs1.append(b)
                graph_indicator.append(graph_id)
                node_labels.append([node_label, activity_map[node_label]])
                node_id += 1
                cnt     += 1
            else:
                img1[i][j] = 0
  
    for i in range(dim1):
        for j in range(dim2):
            #forming edge between all adjacent pixels which are node
            if(nodes[i][j] != -1):
                li = max(0, i - 1)
                ri = min(i + 2, dim1)
                lj = max(0, j - 1)
                rj = min(j + 2, dim2)
                for i1 in range(li, ri):
                    for j1 in range(lj, rj):
                        if((i1 != i or j1 != j) and (nodes[i1][j1] != -1)):
                            edges.append([nodes[i][j],nodes[i1][j1]])
                            edge += 1
  
    attrs1=normalize(attrs1)
    attrs.extend(attrs1)
    del attrs1
    print("For given image nodes formed: " + str(cnt)+" edges formed: " + str(edge))
    if(cnt != 0): 
        graph_id += 1

#generate graphs for all edge-image under given dir along with proper label
def generate_graph_with_labels(dirname, label, activity_map):
    print("\n... Reading Directory: " + dirname + " ...\n")
    global graph_labels
    filenames = glob.glob(dirname + '/*.png')
    for filename in filenames:
        generate_graphs(filename, label, activity_map)
        graph_labels.append([label, activity_map[label]])

#generate graphs for all directories
def process_graphs(Epilepsy_dir,
                   No_Epilepsy_dir,
                   activity_map):
    global node_labels, graph_labels
    generate_graph_with_labels(Epilepsy_dir, 1, activity_map)
    generate_graph_with_labels(No_Epilepsy_dir, 2, activity_map)


    print("Processing done")
    print("Total nodes formed: " + str(len(node_labels)) + "Total graphs formed: " + str(len(graph_labels)))

#working directories


Epilepsy_dir = '/users/mac/Downloads/TUH_EEG/Prewitt_v1/Epilepsy'
No_Epilepsy_dir = '/users/mac/Downloads/TUH_EEG/Prewitt_v1/No_Epilepsy'


start = time.time()

#generate_graph_with_labels(BIRAD_0_dir, 1, activity_map)
process_graphs(Epilepsy_dir,
               No_Epilepsy_dir,
               activity_map)

#check all the lengths of globals
#comment if not necessary
print(len(node_labels))
print(len(graph_labels))
print(len(edges))
print(len(attrs))

#create adjacency dataframe
df_A = pd.DataFrame(columns = ["node-1", "node-2"], data = np.array(edges))
print("Shape of edge dataframe: " + str(df_A.shape))
print("\n--summary of dataframe--\n", df_A.head(50))

#create node label dataframe
df_node_label = pd.DataFrame(data = np.array(node_labels), columns=["label", "activity-name"])
print("shape of node-label dataframe: " + str(df_node_label.shape))
print("\n--summary of dataframe--\n", df_node_label)

#create graph label dataframe
df_graph_label = pd.DataFrame(data = np.array(graph_labels), columns = ["label","activity-name"])
print("shape of node-label dataframe: " + str(df_graph_label.shape))
print("\n--summary of dataframe--\n", df_graph_label.head(50))

#create node-attribute dataframe (normalized grayscale value)
df_node_attr = pd.DataFrame(data = np.array(attrs), columns=["gray-val"])
print("shape of node-attribute dataframe: " + str(df_node_attr.shape))
print("\n--summary of dataframe--\n", df_node_attr.head(50))

#create graph-indicator datframe
df_graph_indicator = pd.DataFrame(data = np.array(graph_indicator), columns=["graph-id"])
print("shape of graph-indicator dataframe: " + str(df_graph_indicator.shape))
print("\n--summary of dataframe--\n", df_graph_indicator.head(50))

#omit activity name later for graph-label and node-label
#since GIN model will only accept the label
df_node_label = df_node_label.drop(["activity-name"], axis=1)
print(df_node_label.head(50))

df_graph_label = df_graph_label.drop(["activity-name"], axis=1)
print(df_graph_label.head(50))



def save_dataframe_to_txt(df, filepath):
    df.to_csv(filepath, header=None, index=None, sep=',', mode='w')



#save all the dataframes to .txt file
#path name: .../GraphTrain/dataset/<dataset_name>/raw/<dataset_name>_<type>.txt
# <type>:
# A--> adjancency matrix
#graph_indicator--> graph-ids of all node
#graph_labels--> labels for all graph
#node_attributes--> attribute(s) for all node
#node_labels--> labels for all node

#sourcepath='/home/linh/Downloads/Retino/Retino_Prewitt_v1/raw'
#os.makedirs(sourcepath, exist_ok=False)
#print("The new directory is created!")
#save_dataframe_to_txt(df_A, sourcepath + '/Retino_Prewitt_v1_A.txt')
#save_dataframe_to_txt(df_graph_indicator, sourcepath + '/Retino_Prewitt_v1_graph_indicator.txt')
#save_dataframe_to_txt(df_graph_label, sourcepath + '/Retino_Prewitt_v1_graph_labels.txt')
#save_dataframe_to_txt(df_node_attr, sourcepath + '/Retino_Prewitt_v1_node_attributes.txt')
#save_dataframe_to_txt(df_node_label, sourcepath + '/Retino_Prewitt_v1_node_labels.txt')


sourcepath='/users/mac/Downloads/TUH_EEG/EEG_Prewitt_v1/raw'
os.makedirs(sourcepath, exist_ok=False)
print("The new directory is created!")
save_dataframe_to_txt(df_A, sourcepath + '/EEG_Prewitt_v1_A.txt')
save_dataframe_to_txt(df_graph_indicator, sourcepath + '/EEG_Prewitt_v1_graph_indicator.txt')
save_dataframe_to_txt(df_graph_label, sourcepath + '/EEG_Prewitt_v1_graph_labels.txt')
save_dataframe_to_txt(df_node_attr, sourcepath + '/EEG_Prewitt_v1_node_attributes.txt')
save_dataframe_to_txt(df_node_label, sourcepath + '/EEG_Prewitt_v1_node_labels.txt')

end = time.time()
time_to_construct = (end - start)/60
print("******* Total time (min) for constructing Graph: ", time_to_construct)
print("======= End constructing Graph process here =======")
