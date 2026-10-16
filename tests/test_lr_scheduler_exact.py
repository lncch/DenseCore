"""torch.optim.lr_scheduler on DenseCore optimizers: same learning rates (and momentum/betas)
every step, and parameters that stay bit-identical to PyTorch's."""
import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')


def _same(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def _schedulers(L, opt):
    return {
        "LambdaLR": lambda: L.LambdaLR(opt, lambda e: 0.95 ** e),
        "MultiplicativeLR": lambda: L.MultiplicativeLR(opt, lambda e: 0.9),
        "StepLR": lambda: L.StepLR(opt, step_size=3, gamma=0.5),
        "MultiStepLR": lambda: L.MultiStepLR(opt, milestones=[2, 5, 7], gamma=0.3),
        "ConstantLR": lambda: L.ConstantLR(opt, factor=0.25, total_iters=4),
        "LinearLR": lambda: L.LinearLR(opt, start_factor=0.1, total_iters=6),
        "ExponentialLR": lambda: L.ExponentialLR(opt, gamma=0.87),
        "PolynomialLR": lambda: L.PolynomialLR(opt, total_iters=8, power=2.0),
        "CosineAnnealingLR": lambda: L.CosineAnnealingLR(opt, T_max=5, eta_min=1e-4),
        "CosineAnnealingWarmRestarts": lambda: L.CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=2),
        "CyclicLR": lambda: L.CyclicLR(opt, base_lr=1e-3, max_lr=1e-2, step_size_up=3, mode="triangular2"),
        "OneCycleLR": lambda: L.OneCycleLR(opt, max_lr=0.05, total_steps=12),
        "SequentialLR": lambda: L.SequentialLR(opt, [L.LinearLR(opt, 0.2, total_iters=3), L.ExponentialLR(opt, 0.9)], milestones=[3]),
        "ChainedScheduler": lambda: L.ChainedScheduler([L.ConstantLR(opt, 0.5, total_iters=4), L.ExponentialLR(opt, 0.9)]),
        "ReduceLROnPlateau": lambda: L.ReduceLROnPlateau(opt, patience=1, factor=0.5),
    }


@pytest.mark.mac_exact
@pytest.mark.parametrize("opt_name", ["SGD", "AdamW"])
@pytest.mark.parametrize("name", list(_schedulers(None, None)))
def test_scheduler_matches_torch(name, opt_name):
    kwargs = dict(lr=0.01, momentum=0.9) if opt_name == "SGD" else dict(lr=0.01)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 5)).astype(np.float32)
    tp, dp = torch.nn.Parameter(torch.tensor(x)), d.tensor(x, requires_grad=True)
    to, do = getattr(torch.optim, opt_name)([tp], **kwargs), getattr(d.optim, opt_name)([dp], **kwargs)
    ts, ds = _schedulers(torch.optim.lr_scheduler, to)[name](), _schedulers(d.optim.lr_scheduler, do)[name]()
    hyper = "momentum" if opt_name == "SGD" else "betas"
    for step in range(12):
        g = rng.normal(size=x.shape).astype(np.float32)
        tp.grad, dp.grad = torch.tensor(g), d.tensor(g)
        to.step(); do.step()
        if name == "ReduceLROnPlateau":
            metric = [3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5][step]
            ts.step(metric); ds.step(metric)
        else:
            ts.step(); ds.step()
        assert ts.get_last_lr() == ds.get_last_lr()
        assert to.param_groups[0]["lr"] == do.param_groups[0]["lr"]
        assert to.param_groups[0][hyper] == do.param_groups[0][hyper]
        assert _same(tp.detach().numpy(), dp.numpy())


def test_scheduler_names_match_torch():
    assert sorted(d.optim.lr_scheduler.__all__) == sorted(torch.optim.lr_scheduler.__all__)


def test_state_dict_round_trip():
    p = d.tensor(np.ones(3, np.float32), requires_grad=True)
    opt = d.optim.SGD([p], lr=0.1)
    s = d.optim.lr_scheduler.StepLR(opt, step_size=2)
    for _ in range(3):
        opt.step(); s.step()
    s2 = d.optim.lr_scheduler.StepLR(d.optim.SGD([p], lr=0.1), step_size=2)
    s2.load_state_dict(s.state_dict())
    assert s2.last_epoch == 3 and s2.get_last_lr() == s.get_last_lr()
