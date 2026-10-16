def test_public_import():
    from densecore import tensor
    assert tensor([1.0, 2.0]).reshape(1, 2).shape == (1, 2)
