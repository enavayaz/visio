# Visio

A general-purpose image classification library built on TensorFlow/Keras.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from visio import ImNet
from visio.data_loader import load_mnist_dataset, split
from visio.utils import make_submission

X_train_full, y_train_full, X_test, y_test = load_mnist_dataset()
X_train, X_val, y_train, y_val = split(X_train_full, y_train_full)

model = ImNet(input_shape=(28, 28, 1), num_classes=10)
model.compile()
model.fit(X_train, y_train, X_val, y_val)

predictions = model.predict(X_test)
make_submission(predictions)
```

## Project Structure

```
visio/
├── visio/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── utils.py
│   └── data_loader.py
├── notebooks/
│   └── run_imnet.ipynb
├── tests/
│   ├── test_cnn.py
│   └── test_utils.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Expected Accuracy on MNIST

| Setup | Accuracy |
|---|---|
| Default (50 epochs) | ~99.5% |
| With GPU | ~99.6% |

## License
MIT
