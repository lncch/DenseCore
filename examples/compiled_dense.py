"""Train a compiled dense module, then reuse its inference workspace."""
import numpy as np
import densecore as dc


def main():
    rng = np.random.default_rng(37)
    values = rng.normal(size=(64, 2)).astype(np.float32)
    targets = (3 * values[:, :1] - 2 * values[:, 1:] + .5).astype(np.float32)
    x, y = dc.tensor(values), dc.tensor(targets)
    dc.manual_seed(7)
    model = dc.compile(dc.nn.Linear(2, 1))
    optimizer = dc.optim.SGD(model.parameters(), lr=.08)
    for _ in range(100):
        optimizer.zero_grad()
        loss = dc.functional.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    with dc.no_grad():
        mse = float(dc.functional.mse_loss(model(x), y).numpy())
    assert mse < 1e-3, f"compiled regression MSE too high: {mse}"
    print(f"Compiled regression MSE: {mse:.8g}")


if __name__ == '__main__':
    main()
