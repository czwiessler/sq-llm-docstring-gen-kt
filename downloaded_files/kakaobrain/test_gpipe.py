from collections import OrderedDict
from copy import deepcopy
import time

import pytest
import torch
from torch import nn

from torchgpipe import GPipe
from torchgpipe.gpipe import verify_module


def test_parameters():
    model = nn.Sequential(nn.Linear(1, 1))
    gpipe = GPipe(model, balance=[1], devices=['cpu'], chunks=1)
    assert list(gpipe.parameters()) != []


def test_public_attrs():
    class MyString:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return self.value

    model = nn.Sequential(nn.Linear(1, 1))
    gpipe = GPipe(model,
                  balance=(1,),
                  devices=('cpu',),
                  chunks=42.000,
                  checkpoint=MyString('always'))

    assert gpipe.balance == [1]
    assert gpipe.devices == [torch.device('cpu')]
    assert gpipe.chunks == 42
    assert isinstance(gpipe.chunks, int)
    assert gpipe.checkpoint == 'always'
    assert isinstance(gpipe.checkpoint, str)


@pytest.mark.parametrize('balance', [[2], [1, 1]])
def test_sequential_like(balance):
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = GPipe(model, balance, devices=['cpu', 'cpu'])

    assert len(model) == 2
    assert list(model) == [a, b]

    assert model[0] is a
    assert model[1] is b
    with pytest.raises(IndexError):
        _ = model[2]

    assert model[-1] is b
    assert model[-2] is a


def test_balance_wrong_length():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        GPipe(model, balance=[1])

    with pytest.raises(ValueError):
        GPipe(model, balance=[3])


def test_balance_less_than_1():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)

    with pytest.raises(ValueError):
        GPipe(model, balance=[0, 2])

    with pytest.raises(ValueError):
        GPipe(model, balance=[-1, 3])


def test_chunks_less_than_1():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError):
        GPipe(model, balance=[1], devices=['cpu'], chunks=0)

    with pytest.raises(ValueError):
        GPipe(model, balance=[1], devices=['cpu'], chunks=-1)


def test_too_few_devices():
    model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1), nn.Linear(1, 1))

    with pytest.raises(IndexError):
        # len(balance) > len(devices)
        model = GPipe(model, balance=[1, 1, 1, 1], devices=['cpu'])


def test_batch_size_indivisible():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(7, 1))

    # Indivisible batch size is legal.
    assert not record


def test_batch_size_small():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=4)

    with pytest.warns(None) as record:
        model(torch.rand(2, 1))

    # Batch size smaller than chunks is legal.
    assert not record


def test_checkpoint_mode():
    def count_grad_fn(grad_fn, name, visited=set()):
        if grad_fn in visited:
            return 0
        visited.add(grad_fn)

        if grad_fn is None:
            return 0
        if grad_fn.__class__.__name__ == name:
            return 1

        counter = 0
        for next_grad_fn, _ in grad_fn.next_functions:
            counter += count_grad_fn(next_grad_fn, name, visited=visited)
        return counter

    model = nn.Sequential(nn.Linear(1, 1))
    input = torch.rand(2, 1)

    always = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='always')
    except_last = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='except_last')
    never = GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='never')

    always_output = always(input)
    except_last_output = except_last(input)
    never_output = never(input)

    assert count_grad_fn(always_output.grad_fn, 'CheckpointBackward') == 2
    assert count_grad_fn(except_last_output.grad_fn, 'CheckpointBackward') == 1
    assert count_grad_fn(never_output.grad_fn, 'CheckpointBackward') == 0


def test_checkpoint_mode_invalid():
    model = nn.Sequential(nn.Linear(1, 1))

    with pytest.raises(ValueError,
                       match="checkpoint is not one of 'always', 'except_last', or 'never'"):
        GPipe(model, balance=[1], devices=['cpu'], chunks=2, checkpoint='INVALID_CHECKPOINT')


def test_checkpoint_mode_when_chunks_1():
    model = nn.Sequential(nn.Linear(1, 1))

    # All checkpoint modes are fine.
    GPipe(model, balance=[1], devices=['cpu'], chunks=1, checkpoint='except_last')
    GPipe(model, balance=[1], devices=['cpu'], chunks=1, checkpoint='always')
    GPipe(model, balance=[1], devices=['cpu'], chunks=1, checkpoint='never')


def test_checkpoint_eval():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)
    input = torch.rand(2, 1)

    def find_grad_fn(grad_fn, name):
        if grad_fn is None:
            return False
        if grad_fn.__class__.__name__ == name:
            return True
        for next_grad_fn, _ in grad_fn.next_functions:
            if find_grad_fn(next_grad_fn, name):
                return True
        return False

    model.train()
    train_output = model(input)
    assert find_grad_fn(train_output.grad_fn, 'CheckpointBackward')
    assert find_grad_fn(train_output.grad_fn, 'RecomputeBackward')

    model.eval()
    eval_output = model(input)
    assert not find_grad_fn(eval_output.grad_fn, 'CheckpointBackward')
    assert not find_grad_fn(eval_output.grad_fn, 'RecomputeBackward')


def test_no_grad():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)
    input = torch.rand(2, 1)

    latent = None

    def hook(module, input, output):
        _ = module
        _ = input

        nonlocal latent
        latent = output

    partition = model.partitions[0]
    partition.register_forward_hook(hook)

    with torch.no_grad():
        model(input)

    assert latent.grad_fn is None


def test_exception():
    class ExpectedException(Exception):
        pass

    class Raise(nn.Module):
        def forward(self, *_):
            raise ExpectedException()

    model = nn.Sequential(Raise())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=1)

    with pytest.raises(ExpectedException):
        model(torch.rand(1))


def test_exception_early_stop_asap():
    """Even the first partitions have finished to process, the partition before
    the failed partition should be killed as soon as possible.
    """
    class ExpectedException(Exception):
        pass

    class Pass(nn.Module):
        def forward(self, x):
            return x

    counter = 0

    class Counter(nn.Module):
        def forward(self, x):
            time.sleep(0.1)

            nonlocal counter
            counter += 1

            return x

    class Raise(nn.Module):
        def forward(self, x):
            raise ExpectedException()

    model = nn.Sequential(Pass(), Pass(), Counter(), Raise())
    model = GPipe(model, [1, 1, 1, 1], devices=['cpu', 'cpu', 'cpu', 'cpu'], chunks=3)

    with pytest.raises(ExpectedException):
        model(torch.rand(3))

    # If the early stop doesn't work, it would be 3 instead.
    assert counter == 2


def test_input_pair():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_a = nn.Linear(1, 1)
            self.fc_b = nn.Linear(1, 1)

        def forward(self, a_and_b):
            a, b = a_and_b
            return (self.fc_a(a), self.fc_b(b))

    model = nn.Sequential(Two())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)

    a = torch.rand(10, 1, requires_grad=True)
    b = torch.rand(10, 1, requires_grad=True)

    a_out, b_out = model((a, b))
    loss = (a_out + b_out).mean()
    loss.backward()

    assert a.grad is not None
    assert b.grad is not None


def test_input_singleton():
    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, only_a):
            a, = only_a
            return (self.fc(a),)

    model = nn.Sequential(One())
    model = GPipe(model, balance=[1], devices=['cpu'], chunks=2)

    a = torch.rand(10, 1, requires_grad=True)

    a_out, = model((a,))
    loss = a_out.mean()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters())
    assert a.grad is not None


def test_input_varargs():
    model = nn.Sequential(nn.Linear(1, 1))
    model = GPipe(model, balance=[1], devices=['cpu'])

    a = torch.rand(1)
    b = torch.rand(1)

    # TypeError: forward() takes 2 positional arguments but 3 were given
    with pytest.raises(TypeError):
        model(a, b)


def test_non_tensor():
    class NonTensor(nn.Module):
        def forward(self, _):
            return 'hello'

    model = nn.Sequential(NonTensor())
    model = GPipe(model, balance=[1], devices=['cpu'])
    x = torch.rand(1)

    # TypeError: expected Tensor as element 0 in argument 0, but got str
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model('hello')


def test_non_tensor_tuple():
    class NonTensorTuple(nn.Module):
        def forward(self, x):
            return (x, 'hello')

    model = nn.Sequential(NonTensorTuple())
    model = GPipe(model, balance=[1], devices=['cpu'])
    x = torch.rand(1)

    # TypeError: CheckpointBackward.forward: expected Variable (got str) for return value 1
    with pytest.raises(TypeError):
        model(x)

    # TypeError: expected Tensor to scatter, but got str
    with pytest.raises(TypeError):
        model((x, 'hello'))


@pytest.mark.parametrize('checkpoint', ['never', 'always', 'except_last'])
def test_deferred_batch_norm(checkpoint):
    bn = nn.BatchNorm2d(3)
    gpipe_bn = deepcopy(bn)
    gpipe = GPipe(nn.Sequential(gpipe_bn), balance=[1], devices=['cpu'], chunks=2,
                  checkpoint=checkpoint, deferred_batch_norm=True)

    x = torch.rand(4, 3, 10, 10)
    gpipe(x).mean().backward()
    bn(x).mean().backward()

    assert torch.allclose(gpipe[0].running_mean, bn.running_mean, atol=1e-4)
    assert torch.allclose(gpipe[0].running_var, bn.running_var, atol=1e-4)


@pytest.mark.parametrize('checkpoint', ['never', 'always'])
def test_deferred_batch_norm_params(checkpoint):
    bn = nn.BatchNorm2d(3)
    gpipe_bn = deepcopy(bn)
    gpipe = GPipe(nn.Sequential(gpipe_bn), balance=[1], devices=['cpu'], chunks=1,
                  checkpoint=checkpoint, deferred_batch_norm=True)

    x = torch.rand(4, 3, 10, 10)
    gpipe(x).mean().backward()
    bn(x).mean().backward()

    assert gpipe[0].weight.grad is not None
    assert gpipe[0].bias.grad is not None

    assert torch.allclose(gpipe[0].weight.grad, bn.weight.grad, atol=1e-4)
    assert torch.allclose(gpipe[0].bias.grad, bn.bias.grad, atol=1e-4)


def test_devices():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)
    c = nn.Linear(1, 1)

    # There are extra two devices.
    devices = ['cpu', 'cpu', 'cpu', 'cpu', 'cpu']

    model = nn.Sequential(a, b, c)
    model = GPipe(model, [1, 1, 1], devices=devices)

    cpu = torch.device('cpu')
    # Extra devices must be discarded.
    assert model.devices == [cpu, cpu, cpu]


def test_partitions():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = GPipe(model, [1, 1], devices=['cpu', 'cpu'])

    assert isinstance(model.partitions, nn.ModuleList)
    assert isinstance(model.partitions[0], nn.Sequential)
    assert isinstance(model.partitions[1], nn.Sequential)

    assert 'partitions.0.0.weight' in model.state_dict()


def test_deny_moving():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(a, b)
    model = GPipe(model, [1, 1], devices=['cpu', 'cpu'])

    # Moving is denied.
    with pytest.raises(TypeError):
        model.cuda()

    with pytest.raises(TypeError):
        model.cpu()

    with pytest.raises(TypeError):
        model.to(torch.device('cuda'))

    with pytest.raises(TypeError):
        model.to(0)

    with pytest.raises(TypeError):
        model.to('cuda')

    with pytest.raises(TypeError):
        model.to(device=0)

    with pytest.raises(TypeError):
        model.to(torch.rand(1))

    with pytest.raises(TypeError):
        model.to(tensor=torch.rand(1))

    # Casting is allowed.
    model.half()
    model.to(torch.double)
    model.to(dtype=torch.float)


def test_empty_module():
    # Empty sequential module is not illegal.
    model = nn.Sequential()
    model = GPipe(model, [])

    assert model(torch.tensor(42)) == torch.tensor(42)
    assert model((torch.tensor(42),)) == (torch.tensor(42),)

    # But only tensor or tensors is legal in GPipe.
    with pytest.raises(TypeError):
        model(42)


def test_named_children():
    a = nn.Linear(1, 1)
    b = nn.Linear(1, 1)

    model = nn.Sequential(OrderedDict([('a', a), ('b', b)]))
    model = GPipe(model, [1, 1], devices=['cpu', 'cpu'])

    names = set(n for n, _ in model.named_modules())
    assert 'partitions.0.a' in names
    assert 'partitions.1.b' in names

    # GPipe doesn't support __getattr__. Unlike nn.Sequential, GPipe requires
    # several methods in its namespace.
    with pytest.raises(AttributeError):
        model.a


def test_recommend_auto_balance():
    with pytest.raises(ValueError, match='torchgpipe.balance'):
        # balance is required
        GPipe(nn.Sequential())

    with pytest.raises(ValueError, match='torchgpipe.balance'):
        # module and sum of balance have differen length (module: 0, sum of balance: 1)
        GPipe(nn.Sequential(), [1])

    with pytest.raises(ValueError, match='torchgpipe.balance'):
        # module and sum of balance have different length (module: 2, sum of balance: 1)
        GPipe(nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)), [1])


def test_verify_module_non_sequential():
    with pytest.raises(TypeError, match='module must be nn.Sequential to be partitioned'):
        verify_module(nn.Module())


def test_verify_module_duplicate_children():
    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(conv, conv)

    with pytest.raises(ValueError, match='module with duplicate children is not supported'):
        verify_module(model)


def test_verify_module_duplicate_parameters_in_distinct_children():
    class Surrogate(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    conv = nn.Conv2d(3, 3, 1)
    model = nn.Sequential(Surrogate(conv), Surrogate(conv))

    with pytest.raises(ValueError, match='module with duplicate parameters in '
                                         'distinct children is not supported'):
        verify_module(model)
