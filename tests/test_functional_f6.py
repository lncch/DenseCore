import numpy as np
import pytest
import densecore as d

torch = pytest.importorskip('torch')
TF = torch.nn.functional
DF = d.functional
rng = np.random.default_rng(0)
# Includes the kink points PyTorch's backward rules care about.
EDGES = np.array([-30., -3.5, -3., -1., -0.5, 0., 0.5, 1., 3., 3.5, 6., 7., 25.], np.float32)


def compare(ours, theirs, inputs, rtol=1e-5):
    np.testing.assert_allclose(ours.numpy(), theirs.detach().numpy(), rtol=rtol, atol=1e-6, equal_nan=True)
    w = rng.normal(size=tuple(theirs.shape)).astype(np.float32)
    (ours * d.tensor(w)).sum().backward()
    (theirs * torch.from_numpy(w)).sum().backward()
    for a, b in inputs:
        if b.grad is None:  # input not on PyTorch's graph: it must not get a gradient here either
            assert a.grad is None or not a.grad.numpy().any()
            continue
        np.testing.assert_allclose(a.grad.numpy(), b.grad.numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


ACTIVATIONS = [('relu', {}), ('relu6', {}), ('hardtanh', {'min_val': -2., 'max_val': 3.}),
               ('leaky_relu', {'negative_slope': 0.2}), ('elu', {'alpha': 1.5}), ('celu', {'alpha': 0.7}),
               ('selu', {}), ('silu', {}), ('mish', {}), ('softplus', {'beta': 2., 'threshold': 10.}),
               ('softsign', {}), ('softshrink', {'lambd': 1.}), ('hardshrink', {'lambd': 1.}),
               ('hardsigmoid', {}), ('hardswish', {}), ('logsigmoid', {}), ('tanhshrink', {}),
               ('threshold', {'threshold': 0.5, 'value': -9.}), ('softmin', {'dim': 0}),
               ('glu', {'dim': 0}), ('normalize', {'dim': 0})]


@pytest.mark.parametrize('name,kwargs', ACTIVATIONS)
def test_activations_match_torch_at_kinks(name, kwargs):
    values = EDGES[:12] if name == 'glu' else EDGES
    x, tx = d.tensor(values, requires_grad=True), torch.tensor(values, requires_grad=True)
    compare(getattr(DF, name)(x, **kwargs), getattr(TF, name)(tx, **kwargs), [(x, tx)])


def test_parametric_and_shape_activations():
    x, tx = d.tensor(rng.normal(size=(2, 3, 4)).astype(np.float32), requires_grad=True), None
    tx = torch.tensor(x.numpy(), requires_grad=True)
    w, tw = d.tensor(np.array([0.1, 0.2, 0.3], np.float32), requires_grad=True), torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
    compare(DF.prelu(x, w), TF.prelu(tx, tw), [(x, tx), (w, tw)])
    y = rng.normal(size=(1, 8, 2, 3)).astype(np.float32)
    for f, g in [(lambda t: DF.pixel_shuffle(t, 2), lambda t: TF.pixel_shuffle(t, 2)),
                 (lambda t: DF.channel_shuffle(t, 4), lambda t: TF.channel_shuffle(t, 4))]:
        a, ta = d.tensor(y, requires_grad=True), torch.tensor(y, requires_grad=True)
        compare(f(a), g(ta), [(a, ta)])
    z = rng.normal(size=(1, 2, 4, 6)).astype(np.float32)
    a, ta = d.tensor(z, requires_grad=True), torch.tensor(z, requires_grad=True)
    compare(DF.pixel_unshuffle(a, 2), TF.pixel_unshuffle(ta, 2), [(a, ta)])
    a1, a2 = rng.normal(size=(4, 3)).astype(np.float32), rng.normal(size=(4, 5)).astype(np.float32)
    bw, bb = rng.normal(size=(2, 3, 5)).astype(np.float32), rng.normal(size=2).astype(np.float32)
    ins = [(d.tensor(v, requires_grad=True), torch.tensor(v, requires_grad=True)) for v in (a1, a2, bw, bb)]
    compare(DF.bilinear(*(p[0] for p in ins)), TF.bilinear(*(p[1] for p in ins)), ins)
    r = rng.normal(size=(3, 4)).astype(np.float32)
    a, ta = d.tensor(r, requires_grad=True), torch.tensor(r, requires_grad=True)
    compare(DF.rms_norm(a, 4), TF.rms_norm(ta, (4,)), [(a, ta)])
    p, q = (rng.normal(size=(5, 3)).astype(np.float32) for _ in range(2))
    ins = [(d.tensor(v, requires_grad=True), torch.tensor(v, requires_grad=True)) for v in (p, q)]
    compare(DF.cosine_similarity(ins[0][0], ins[1][0]), TF.cosine_similarity(ins[0][1], ins[1][1]), ins)
    ins = [(d.tensor(v, requires_grad=True), torch.tensor(v, requires_grad=True)) for v in (p, q)]
    compare(DF.pairwise_distance(ins[0][0], ins[1][0]), TF.pairwise_distance(ins[0][1], ins[1][1]), ins)
    a, ta = d.tensor(p, requires_grad=True), torch.tensor(p, requires_grad=True)
    compare(DF.pdist(a), TF.pdist(ta), [(a, ta)])


PRED = rng.uniform(0.05, 0.95, size=(4, 3)).astype(np.float32)
TARGET = rng.uniform(0, 1, size=(4, 3)).astype(np.float32)
LOGITS = np.array([[0., -2., 3.], [40., -40., 0.5], [1., 1., -1.], [0., 0., 0.]], np.float32)
SIGNS = np.array([1., -1., 1., -1.], np.float32)
LOSSES = [
    ('l1_loss', lambda F, x, t: F.l1_loss(x, t)), ('l1_loss', lambda F, x, t: F.l1_loss(x, t, reduction='none')),
    ('smooth_l1', lambda F, x, t: F.smooth_l1_loss(x * 4, t, beta=0.5)), ('huber', lambda F, x, t: F.huber_loss(x * 4, t, delta=0.7)),
    ('bce', lambda F, x, t: F.binary_cross_entropy(x, t)), ('bce_sum', lambda F, x, t: F.binary_cross_entropy(x, t, reduction='sum')),
    ('kl', lambda F, x, t: F.kl_div(x.log(), t, reduction='batchmean')), ('kl_log', lambda F, x, t: F.kl_div(x, t.log(), reduction='batchmean', log_target=True)),
    ('poisson', lambda F, x, t: F.poisson_nll_loss(x, t)), ('poisson_full', lambda F, x, t: F.poisson_nll_loss(x, t * 4, log_input=False, full=True)),
    ('gaussian', lambda F, x, t: F.gaussian_nll_loss(x, t, x * 0.5)), ('soft_margin', lambda F, x, t: F.soft_margin_loss(x, t.detach() * 2 - 1)),
    ('multilabel_soft', lambda F, x, t: F.multilabel_soft_margin_loss(x, (t > 0.5).float())),
    ('triplet', lambda F, x, t: F.triplet_margin_loss(x, t, x * 0.3, margin=0.5)),
]


@pytest.mark.parametrize('name,f', LOSSES)
def test_losses_match_torch(name, f):
    x, tx = d.tensor(PRED, requires_grad=True), torch.tensor(PRED, requires_grad=True)
    t, tt = d.tensor(TARGET, requires_grad=True), torch.tensor(TARGET, requires_grad=True)
    compare(f(DF, x, t), f(TF, tx, tt), [(x, tx), (t, tt)])


def test_bce_with_logits_and_ranking_losses():
    x, tx = d.tensor(LOGITS, requires_grad=True), torch.tensor(LOGITS, requires_grad=True)
    t = (TARGET > 0.5).astype(np.float32)
    pw = np.array([1., 2., 0.5], np.float32)
    compare(DF.binary_cross_entropy_with_logits(x, d.tensor(t), pos_weight=d.tensor(pw)),
            TF.binary_cross_entropy_with_logits(tx, torch.from_numpy(t), pos_weight=torch.from_numpy(pw)), [(x, tx)])
    a, ta = d.tensor(LOGITS[:, 0], requires_grad=True), torch.tensor(LOGITS[:, 0], requires_grad=True)
    b, tb = d.tensor(LOGITS[:, 2], requires_grad=True), torch.tensor(LOGITS[:, 2], requires_grad=True)
    compare(DF.margin_ranking_loss(a, b, d.tensor(SIGNS), 0.5), TF.margin_ranking_loss(ta, tb, torch.from_numpy(SIGNS), 0.5), [(a, ta), (b, tb)])
    a, ta = d.tensor(LOGITS[:, 2], requires_grad=True), torch.tensor(LOGITS[:, 2], requires_grad=True)
    compare(DF.hinge_embedding_loss(a, d.tensor(SIGNS)), TF.hinge_embedding_loss(ta, torch.from_numpy(SIGNS)), [(a, ta)])
    p, q = d.tensor(LOGITS[:, :2] + 0.1, requires_grad=True), None
    tp = torch.tensor(LOGITS[:, :2] + 0.1, requires_grad=True)
    r, tr = d.tensor(TARGET[:, :2], requires_grad=True), torch.tensor(TARGET[:, :2], requires_grad=True)
    compare(DF.cosine_embedding_loss(p, r, d.tensor(SIGNS), 0.2), TF.cosine_embedding_loss(tp, tr, torch.from_numpy(SIGNS), 0.2), [(p, tp), (r, tr)])
    labels = np.array([2, 0, 1, 1])
    for kwargs in ({}, {'p': 2, 'margin': 0.5}, {'weight': np.array([1., 2., 3.], np.float32)}):
        x, tx = d.tensor(LOGITS / 10, requires_grad=True), torch.tensor(LOGITS / 10, requires_grad=True)
        dk = {k: d.tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        tk = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        compare(DF.multi_margin_loss(x, d.tensor(labels, dtype='int64'), **dk),
                TF.multi_margin_loss(tx, torch.from_numpy(labels), **tk), [(x, tx)])


@pytest.mark.parametrize('kwargs', [{}, {'reduction': 'sum'}, {'reduction': 'none'}, {'label_smoothing': 0.2},
                                    {'weight': np.array([1., 3., .5], np.float32)}, {'ignore_index': 1},
                                    {'weight': np.array([1., 3., .5], np.float32), 'label_smoothing': 0.1, 'ignore_index': 0}])
def test_cross_entropy_and_nll_variants_match_torch(kwargs):
    dk = {k: d.tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    tk = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    logits = rng.normal(size=(2, 3, 4)).astype(np.float32)  # N, C, d1
    labels = np.array([[0, 1, 2, 1], [2, 2, 0, 1]])
    x, tx = d.tensor(logits, requires_grad=True), torch.tensor(logits, requires_grad=True)
    compare(DF.cross_entropy(x, d.tensor(labels, dtype='int64'), **dk), TF.cross_entropy(tx, torch.from_numpy(labels), **tk), [(x, tx)])
    if 'label_smoothing' not in kwargs:
        x, tx = d.tensor(logits, requires_grad=True), torch.tensor(logits, requires_grad=True)
        compare(DF.nll_loss(DF.log_softmax(x, 1), d.tensor(labels, dtype='int64'), **dk),
                TF.nll_loss(TF.log_softmax(tx, 1), torch.from_numpy(labels), **tk), [(x, tx)])
    probs = np.abs(rng.normal(size=(2, 3, 4))).astype(np.float32)
    probs /= probs.sum(1, keepdims=True)
    if 'ignore_index' not in kwargs:
        x, tx = d.tensor(logits, requires_grad=True), torch.tensor(logits, requires_grad=True)
        compare(DF.cross_entropy(x, d.tensor(probs), **dk), TF.cross_entropy(tx, torch.from_numpy(probs), **tk), [(x, tx)])


def test_modules_wrap_functionals():
    x = d.tensor(EDGES)
    assert np.allclose(d.nn.LeakyReLU(0.2)(x).numpy(), TF.leaky_relu(torch.from_numpy(EDGES), 0.2).numpy())
    assert np.allclose(d.nn.Softplus(beta=2.)(x).numpy(), TF.softplus(torch.from_numpy(EDGES), beta=2.).numpy())
    assert d.nn.Identity()(x) is x and d.nn.PReLU(3).weight.shape == (3,)
    with pytest.raises(TypeError):
        d.nn.ELU(bogus=1)
    loss = d.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
    assert loss(d.tensor(LOGITS), d.tensor([0, 1, 2, 0], dtype='int64')).shape == ()
