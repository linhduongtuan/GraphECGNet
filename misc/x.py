import os
import time
import logging
import numpy as np
import torch
from torchvision import transforms, datasets

logger = logging.getLogger('inference')

import os,  time, itertools
import numpy as np
import scikitplot as skplt
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from timm import create_model
torch.backends.cudnn.benchmark = True

data_dir = '/home/linh/Downloads/Mammograms/'
batch_size = 64
img_size=800
test_size = int((256 / 224) * img_size)
mean = [0.485, 0.456, 0.406] 
std = [0.229, 0.224, 0.225]
num_workers = 4
device = torch.device("cuda")

# Define your transforms for the training and testing sets
data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(test_size),
        transforms.CenterCrop(img_size),
        transforms.ColorJitter(0.5, 0.5, 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['test']}
class_names = image_datasets['test'].classes
num_classes = len(class_names)
data_loader = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers, pin_memory = True)
              for x in ['test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}

print(class_names)
print(dataset_sizes)
print(device)

### we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
_ = image_datasets['test'].class_to_idx
cat_to_name = {_[i]: i for i in list(_.keys())}
print(cat_to_name)
    
# Run this to test the data loader
images, labels = next(iter(data_loader['test']))
images.size()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.suam(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")  
    plt.show()
    
def plt_roc(test_y, probas_y, plot_micro=False, plot_macro=False):
    assert isinstance(test_y, list) and isinstance(probas_y, list), 'the type of input must be list'
    #ROCAUC(test_y, probas_y, plot_micro=plot_micro,plot_macro=plot_macro)
    #ROCAUC(test_y, probas_y, plot_micro=plot_micro,plot_macro=plot_macro).score(test_y, probas_y)
    #ROCAUC(test_y, probas_y, plot_micro=plot_micro,plot_macro=plot_macro).show()

    skplt.metrics.plot_roc(test_y, probas_y, plot_micro=plot_micro,plot_macro=plot_macro) #, figsize=(10, 8))
    #plt.savefig(add_prefix(args.prefix, 'roc_auc_curve.png'))
    plt.show()
    #plt.close()

model = create_model('resnet26D', pretrained=True)
def compute_validate_meter(model, val_loader): # best_model_path,
    
    since = time.time()
    model = create_model(
        'resnet26D',
        num_classes=8,
        in_chans=3,
        pretrained=True,
        checkpoint_path='/home/linh/Downloads/Mammograms/weights/20211119-122040-resnet26d-800/ResNet26D.pth')
    model.to(device)
    model.eval()
    pred_y = list()
    test_y = list()
    probas_y = list()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data, target = Variable(data), Variable(target)
            output = model(data)
            probas_y.extend(output.data.cpu().numpy().tolist())
            pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            test_y.extend(target.data.cpu().numpy().flatten().tolist())
        # compute the confusion matrix
        confusion = confusion_matrix(test_y, pred_y)
        # plot the confusion matrix
        plot_labels = ["BIRAD_0", "BIRAD_1","BIRAD_2","BIRAD_3","BIRAD_4A","BIRAD_4B","BIRAD_4C","BIRAD_5",]
        #plot_labels =['Benign', 'Malignant', 'Negative']
        plot_confusion_matrix(confusion, plot_labels)
        #plot_confusion_matrix(confusion, classes=val_loader.dataset.classes,title='Confusion matrix')
        # print Recall, Precision, F1-score, Accuracy
        report = classification_report(test_y, pred_y, digits=4)
        print(report)
        plt_roc(test_y, probas_y)
        
    time_elapsed = time.time() - since

    print('Inference completes in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
count = count_parameters(model)
print(count)

#best_model_path = '/home/linh/Downloads/TB/weights/EfficientNet_B1_240.pth'

compute_validate_meter(model, data_loader['test']) #best_model_path,