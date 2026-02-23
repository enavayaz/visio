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


def reshape(data, target_shape=None):
    """
    Reshapes data for ML models.
    If target_shape is None, it tries to add a trailing channel dimension.
    """
    if data is None:
        return None

    data = np.asarray(data)

    # CASE 1: User provided a specific shape (e.g., (28, 28, 1) or (784,))
    if target_shape is not None:
        new_shape = (data.shape[0], *target_shape)
        return data.reshape(new_shape)

    # CASE 2: No shape provided - Default to adding a channel dim if it's a 2D image
    # (N, H, W) -> (N, H, W, 1)
    if len(data.shape) == 3:
        return np.expand_dims(data, axis=-1)

    return data

def make_submission(predictions, output_path="submission.csv"):
    # Ensure predictions are a 1D array of integers
    if hasattr(predictions, "flatten"):
        predictions = predictions.flatten()

    submission = pd.DataFrame({
        "ImageId": np.arange(1, len(predictions) + 1),
        "Label": predictions.astype(int)  # Forces clean integer labels
    })

    print("--- Submission Preview ---")
    print(submission.head(10))
    print(f"Shape: {submission.shape}")

    submission.to_csv(output_path, index=False)
    print(f"File saved successfully: {output_path}")
    return submission  # Returning the df can be useful for further checks

def evaluate(model, X_val, y_val, conf_matrix=True):
    """Compute accuracy, confusion matrix and classification report on val set."""

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy : {acc:.4f} ({acc*100:.2f}%)\n")

    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=[str(i) for i in range(10)]))
    if conf_matrix:
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

