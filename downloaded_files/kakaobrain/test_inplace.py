import pytest
import torch
from torch import nn

from torchgpipe import GPipe


def test_inplace_on_requires_grad():
    model = nn.Sequential(nn.Linear(1, 1), nn.ReLU(inplace=True))
    model = GPipe(model, [1, 1], devices=['cpu', 'cpu'], checkpoint='always')

    x = torch.rand(1)
    y = model(x)

    match = 'a leaf Variable that requires grad (is being|has been) used in an in-place operation.'
    with pytest.raises(RuntimeError, match=match):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_on_not_requires_grad():
    # In-place operation on a tensor not requiring grad doesn't cause a
    # RuntimeError. Currently, we cannot detect this case.
    model = nn.Sequential(nn.ReLU(inplace=True))
    model = GPipe(model, [1], devices=['cpu'], checkpoint='always')

    x = torch.rand(1)
    y = model(x)

    match = 'a leaf Variable that requires grad (is being|has been) used in an in-place operation.'
    with pytest.raises(RuntimeError, match=match):
        y.backward()


@pytest.mark.xfail(strict=True)
def test_inplace_incorrect_grad():
    class M(nn.Module):
        def forward(self, foo_bar):
            # 'foo' requires grad but 'bar' does not. In-place operation on
            # 'bar' won't cause a RuntimeError.
            foo, bar = foo_bar

            # add_(1) is not idempotent, in contrast to relu_(). If it is
            # executed multiple times, it will accumulates each difference onto
            # 'bar'.
            bar.add_(1)

            # 'bar' is still captured by checkpointing. 'foo' will get
            # incorrect grad.
            return foo * bar

    model = nn.Sequential(M())
    model = GPipe(model, [1], devices=['cpu'], checkpoint='always')

    foo = torch.tensor([1.], requires_grad=True)
    bar = torch.tensor([1.])

    output = model((foo, bar))
    output.backward()

    # The gradient of 'foo' should be 2, but it is 3 actually because
    # bar.add_(1) was executed twice due to checkpointing.
    assert foo.grad.item() == 2.
