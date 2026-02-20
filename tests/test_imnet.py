from visio import ImNet
import numpy as np

def test_compile():
    model = ImNet(epochs=1)
    model.compile()
    assert model.model is not None
    assert model.datagen is not None

def test_fit_predict():
    model = ImNet(epochs=1, batch_size=32)
    model.compile()
    X = np.random.rand(100, 28, 28, 1)
    y = np.random.randint(0, 10, 100)
    X_val = np.random.rand(20, 28, 28, 1)
    y_val = np.random.randint(0, 10, 20)
    model.fit(X, y, X_val, y_val)
    preds = model.predict(X_val)
    assert len(preds) == 20
    assert set(preds).issubset(set(range(10)))
