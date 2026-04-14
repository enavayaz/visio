# Visio

A general-purpose image classification library built on TensorFlow/Keras.

## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/enavayaz/visio/blob/main/notebooks/run_imnet.ipynb)

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
visio/
├── visio/
│── models/
├── utils/
├── notebooks/
├── saved_models/
├── submissions/
├── tests/
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
