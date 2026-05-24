import numpy as np

def accuracy(preds, labels):
    """
    Metric for MNIST.
    preds : [batch, D]
    labels : [batch, ]
    """
    assert preds.shape[0] == labels.shape[0]

    predict_label = np.argmax(preds, axis=-1)
    
    return (predict_label == labels).sum() / preds.shape[0]


def confusion_matrix(preds, labels, num_classes=10):
    """
    Compute confusion matrix.
    preds: [batch, num_classes]
    labels: [batch,]
    """
    pred_labels = np.argmax(preds, axis=-1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(labels, pred_labels):
        cm[true_label, pred_label] += 1
    
    return cm
