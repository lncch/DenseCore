import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
TF, DF = torch.nn.functional, d.functional
rng = np.random.default_rng(0)


def run(f, *arrays, rtol=1e-4):
    """Compare f(library, *tensors) in both libraries: values and every input's gradient."""
    ours = [d.tensor(a, requires_grad=True) for a in arrays]
    theirs = [torch.tensor(a, requires_grad=True) for a in arrays]
    o, t = f(DF, *ours), f(TF, *theirs)
    o, t = (o[0], t[0]) if isinstance(t, tuple) else (o, t)
    assert o.shape == tuple(t.shape), (o.shape, tuple(t.shape))
    np.testing.assert_allclose(o.numpy(), t.detach().numpy(), rtol=rtol, atol=1e-5)
    w = rng.normal(size=tuple(t.shape)).astype(np.float32)
    (o * d.tensor(w)).sum().backward()
    (t * torch.from_numpy(w)).sum().backward()
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=rtol, atol=1e-4)


X2 = rng.normal(size=(2, 4, 7, 6)).astype(np.float32)
W2 = rng.normal(size=(6, 2, 3, 2)).astype(np.float32)   # groups=2
B6 = rng.normal(size=6).astype(np.float32)


# PyTorch warns (performance only) that padding='same' with an even kernel copies the input.
@pytest.mark.filterwarnings("ignore:Using padding='same' with even kernel lengths")
@pytest.mark.parametrize('kwargs', [{}, {'stride': 2, 'padding': 1}, {'dilation': (2, 1), 'padding': (2, 0)},
                                    {'padding': 'same'}, {'stride': (1, 2), 'padding': 'valid'}])
def test_conv2d_groups_stride_padding_dilation(kwargs):
    run(lambda F, x, w, b: F.conv2d(x, w, b, groups=2, **kwargs), X2, W2, B6)


def test_conv1d_conv3d_and_unbatched():
    run(lambda F, x, w: F.conv1d(x, w, stride=2, padding=1), X2[:, :, :, 0], rng.normal(size=(3, 4, 3)).astype(np.float32))
    run(lambda F, x, w: F.conv3d(x, w, padding=1), X2[:1, :2, :4, :4, None].repeat(3, -1), rng.normal(size=(2, 2, 3, 3, 2)).astype(np.float32))
    run(lambda F, x, w: F.conv2d(x, w), X2[0], rng.normal(size=(3, 4, 2, 2)).astype(np.float32))


@pytest.mark.parametrize('kwargs', [{}, {'stride': 2, 'padding': 1, 'output_padding': 1}, {'dilation': 2}])
def test_conv_transpose2d(kwargs):
    weight = rng.normal(size=(4, 3, 3, 3)).astype(np.float32)
    run(lambda F, x, w: F.conv_transpose2d(x, w, groups=1, **kwargs), X2, weight)
    run(lambda F, x, w: F.conv_transpose1d(x, w[:, :, :, 0], stride=2), X2[:, :, :, 0], weight)


@pytest.mark.parametrize('f', [
    lambda F, x: F.max_pool2d(x, 2), lambda F, x: F.max_pool2d(x, 3, 2, 1, ceil_mode=True),
    lambda F, x: F.max_pool2d(x, (2, 3), dilation=2), lambda F, x: F.max_pool1d(x[:, :, 0], 3, 2),
    lambda F, x: F.avg_pool2d(x, 3, 2, 1), lambda F, x: F.avg_pool2d(x, 3, 2, 1, count_include_pad=False, ceil_mode=True),
    lambda F, x: F.avg_pool2d(x, 2, divisor_override=3), lambda F, x: F.adaptive_avg_pool2d(x, (3, 4)),
    lambda F, x: F.adaptive_max_pool2d(x, (4, 4)), lambda F, x: F.adaptive_avg_pool1d(x[:, :, 0], 5),
    lambda F, x: F.pad(x, (1, 2, 0, 3)), lambda F, x: F.pad(x, (2, 1, 1, 0), 'reflect'),
    lambda F, x: F.pad(x, (1, 1, 2, 2), 'replicate'), lambda F, x: F.pad(x, (3, 0, 0, 2), 'circular'),
    lambda F, x: F.pad(x, (-1, 2, 1, -2), value=0.5),
    lambda F, x: F.interpolate(x, scale_factor=2), lambda F, x: F.interpolate(x, size=(5, 9), mode='nearest-exact'),
    lambda F, x: F.interpolate(x, size=(9, 4), mode='bilinear'), lambda F, x: F.interpolate(x, scale_factor=1.7, mode='bilinear', align_corners=True),
    lambda F, x: F.interpolate(x, size=(8, 8), mode='bicubic'), lambda F, x: F.interpolate(x, size=(3, 2), mode='area'),
    lambda F, x: F.group_norm(x, 2), lambda F, x: F.instance_norm(x), lambda F, x: F.local_response_norm(x, 3),
    lambda F, x: F.unfold(x, (2, 3), dilation=1, padding=1, stride=2), lambda F, x: F.layer_norm(x, (7, 6))])
def test_pooling_padding_interpolation_and_norms(f):
    run(f, X2)


def test_max_pool_indices_unpool_and_fold():
    x, tx = d.tensor(X2), torch.tensor(X2)
    values, indices = DF.max_pool2d(x, 2, return_indices=True)
    tvalues, tindices = TF.max_pool2d(tx, 2, return_indices=True)
    np.testing.assert_array_equal(indices.numpy(), tindices.numpy())
    np.testing.assert_array_equal(DF.max_unpool2d(values, indices, 2).numpy(), TF.max_unpool2d(tvalues, tindices, 2).numpy())
    cols = rng.normal(size=(2, 4 * 4, 4 * 5)).astype(np.float32)  # 5x6 output, 2x2 kernel: 4x5 windows
    run(lambda F, c: F.fold(c, (5, 6), 2, stride=1), cols)


def test_batch_norm_training_and_eval_with_running_stats():
    for training in (True, False):
        mean, var = rng.normal(size=4).astype(np.float32), rng.uniform(0.5, 2, 4).astype(np.float32)
        rm, rv, trm, trv = d.tensor(mean), d.tensor(var), torch.tensor(mean), torch.tensor(var)
        g, b = rng.normal(size=4).astype(np.float32), rng.normal(size=4).astype(np.float32)
        run(lambda F, x, w, bb: F.batch_norm(x, rm if F is DF else trm, rv if F is DF else trv, w, bb, training, 0.3), X2, g, b)
        np.testing.assert_allclose(rm.numpy(), trm.numpy(), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(rv.numpy(), trv.numpy(), rtol=1e-5, atol=1e-6)


def test_modules_with_copied_weights_match_torch():
    d.manual_seed(0)
    pairs = [(d.nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2), torch.nn.Conv2d(4, 6, 3, stride=2, padding=1, groups=2)),
             (d.nn.ConvTranspose2d(4, 3, 2, stride=2), torch.nn.ConvTranspose2d(4, 3, 2, stride=2)),
             (d.nn.BatchNorm2d(4, momentum=0.2), torch.nn.BatchNorm2d(4, momentum=0.2)),
             (d.nn.GroupNorm(2, 4), torch.nn.GroupNorm(2, 4)), (d.nn.MaxPool2d(2), torch.nn.MaxPool2d(2)),
             (d.nn.AdaptiveAvgPool2d(1), torch.nn.AdaptiveAvgPool2d(1)), (d.nn.ReflectionPad2d(1), torch.nn.ReflectionPad2d(1)),
             (d.nn.Upsample(scale_factor=2, mode='bilinear'), torch.nn.Upsample(scale_factor=2, mode='bilinear'))]
    for ours, theirs in pairs:
        state = dict(ours.named_parameters())
        with torch.no_grad():
            for name, p in theirs.named_parameters():
                p.copy_(torch.from_numpy(state[name].numpy()))
        for mode in ('train', 'eval'):
            getattr(ours, mode)(), getattr(theirs, mode)()
            np.testing.assert_allclose(ours(d.tensor(X2)).numpy(), theirs(torch.tensor(X2)).detach().numpy(), rtol=1e-4, atol=1e-5)
    conv = d.nn.Conv2d(4, 6, 3)
    bound = 1 / np.sqrt(4 * 9)
    assert np.abs(conv.weight.numpy()).max() <= bound and conv.bias.shape == (6,)
    bn = d.nn.BatchNorm1d(3)
    bn(d.tensor(rng.normal(size=(5, 3)).astype(np.float32)))
    assert bn.running_mean.numpy().any()
