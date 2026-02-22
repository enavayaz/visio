import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

def normalize(X, max_val=255.0):
    """
    Normalize array values to [0, 1].

    Args:
        X      : Input array.
        max_val: Maximum pixel value to divide by. Default: 255.0.

    Returns:
        Normalized array.
    """
    return X / max_val


def reshape(X, shape=None):
    """
    Reshape array to target shape.

    Args:
        X    : Input array.
        shape: Target shape. If None, infers (N, H, W, 1) from X automatically.

    Returns:
        Reshaped array.
    """
    if shape is None:
        n = X.shape[0]
        hw = int(X.shape[1] ** 0.5)   # infers H and W from flat input
        shape = (n, hw, hw, 1)
    return X.reshape(shape)

def make_submission(predictions, output_path="submission.csv"):
    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return submission

def evaluate(model, X_val, y_val):
    """Compute accuracy, confusion matrix and classification report on val set."""

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy : {acc:.4f} ({acc*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=[str(i) for i in range(10)]))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.show()


    return acc
