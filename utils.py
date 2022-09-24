import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch_geometric.loader import DataLoader

################################################################################
# METRICS


def compute_accuracy(
    model: torch.nn.Module,
    loader: DataLoader,
    device=None
):
    """Compute accuracy of input model over all samples from the loader.
    Args:
        model : torch.nn.Module
            NN model
        loader : DataLoader
            Data loader to evaluate on
        device : torch.device, optional
            Device to use, by default None.
            If None uses cuda if available else cpu.
    Returns:
        float :
            Accuracy in [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    y_preds = []
    y_trues = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_preds.append(out.argmax(dim=1))  # Use the class with highest probability.
        y_trues.append(data.y)  # Check against ground-truth labels.

    y_pred = torch.cat(y_preds).flatten()
    y_true = torch.cat(y_trues).flatten()

    return torch.sum(y_pred == y_true).item() / len(y_true)  # Derive ratio of correct predictions.


def plot_cm(cm, 
            display_labels= ['BIRAD_0', 'BIRAD_1', 'BIRAD_2', 'BIRAD_3','BIRAD_4A', 'BIRAD_4B','BIRAD_4C', 'BIRAD_5']):

    """Plot confusion matrix with heatmap.
    Args:
        cm : array
            Confusion matrix
        display_labels : list, optional
            Labels of classes in confusion matrix, by default ["Mutag", "Non Mutag"]
    """
    # Set fontsize for plots
    font = {"size": 20}
    matplotlib.rc("font", **font)

    # Plot confusion matrix
    f, axes = plt.subplots(1, 1, figsize=(10, 10), sharey="row")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=axes, xticks_rotation=45, cmap="Blues", values_format='d')
    #disp.im_.colorbar.remove()
    disp.ax_.set_xlabel("Predicted label", fontsize=20)
    disp.ax_.set_ylabel("True label", fontsize=20)
    #matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    #plt.savefig(f"/home/linh/Downloads/data/results/Confusion_matrix_" + name_dataset + ".png")
    plt.show()


def compute_cm(model, loader_test, device=None):
    """Compute confusion matrix of input model over all samples from the loader.
    Args:
        model : torch.nn.Module
            NN model
        loader_test : DataLoader
            Data loader
        device : torch.device, optional
            Device to use, by default None.
            If None uses cuda if available else cpu.
    Returns:
        array:
            Confusion matrix
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.no_grad():
        model.eval()
        test_batch = next(iter(loader_test)).to(device)

        y_pred = model(test_batch.x, test_batch.edge_index, test_batch.batch).argmax(dim=1)
        y_true = test_batch.y


    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten().cpu(), y_pred.cpu())

    return cm