import os
import torch
import random
import numpy as np
from itertools import product

def set_seed(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets  = targets.float().to(device)

            # Use **logits, _, _ = model(features)** for GoogLeNet to account for aux outputs
            logits, _, _ = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

            accuracy = (correct_pred.float()/num_examples)*100

    return accuracy

def compute_confusion_matrix(model, data_loader, device):
    all_targets, all_predictions = [], []

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets  = targets

            logits, _, _ = model(features)
            _, predicted_labels = torch.max(logits, 1)
            all_targets.extend(targets.to('cpu'))
            all_predictions.extend(predicted_labels.to('cpu'))

    all_predictions = all_predictions
    all_predictions = np.array(all_predictions)
    all_targets     = np.array(all_targets)

    class_labels = np.unique(np.concatenate((all_targets, all_predictions)))

    if class_labels.shape[0] == 1:
        if class_labels[0] != 0:
            class_labels = np.array([class_labels[0]])
        else:
            class_labels = np.array([class_labels[0], 1])

    n_labels = class_labels.shape[0]

    lst = []
    z   = list(zip(all_targets, all_predictions))
    for combi in product(class_labels, repeat=2):
        lst.append(z.count(combi))

    mat = np.asarray(lst)[:, None].reshape(n_labels, n_labels)
    return mat
