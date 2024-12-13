import shutil
import sys
import os
import glob
import re
import numpy as np
import torch
import importlib

def save(model, optimizer, lr_scheduler, model_dir,
         in_channels, n_classes, epoch, train_args):
    save_dict = {
        'model': model.state_dict(),
        'model_args': {
            'in_channels': in_channels,
            'n_classes': n_classes,
            **train_args.model_args,
        },
        'transformer_args': train_args.transformer_args,
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'train_args': train_args.__dict__
    }
    torch.save(
        save_dict,
        os.path.join(model_dir, 'checkpoint-{}'.format(epoch)))


def prompt_delete_dir(directory):
    if os.path.exists(directory):
        answer = input(
            "{} exists. Do you want to delete it?[y/n]".format(directory))
        if answer == 'y':
            shutil.rmtree(directory)
        elif answer != 'n':
            sys.exit(1)


def restore(checkpoint_path, model, optimizer=None, lr_scheduler=None,
            restore_cpu=False, strict=True, continue_training=False):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, location: storage if restore_cpu else None)
    checkpoint['model'] = {
        re.sub('^module[.]', '', key): value
        for key, value in checkpoint['model'].items()
    }
    if not strict:
        msd = model.state_dict()
        checkpoint['model'] = {
            k: v for k, v in checkpoint['model'].items()
            if k in msd and msd[k].shape == v.shape
        }
        print("Loading {} parameters".format(len(checkpoint['model'])))

    model.load_state_dict(checkpoint['model'], strict=strict)
    if optimizer is not None and continue_training:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if (lr_scheduler is not None and 'lr_scheduler' in checkpoint and
            continue_training):
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    return checkpoint['epoch']


def get_latest_checkpoint(model_dir):
    def key_fn(path):
        matches = re.match(r'checkpoint-(\d+)', os.path.basename(path))
        return int(matches.groups()[0])
    return sorted(
        glob.glob(os.path.join(model_dir, 'checkpoint-*')),
        key=key_fn
    )[-1]


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if isinstance(tensor, torch.autograd.Variable):
        tensor = tensor.data
    return tensor.numpy()


def import_type(name, modules=[]):
    type_paths = [name] + [m + '.' + name for m in modules]
    for tp in type_paths:
        try:
            components = tp.split('.')
            mod = importlib.import_module(components[0])
            for comp in components[1:]:
                mod = getattr(mod, comp)
            return mod
        except:
            pass
    raise ImportError(name + ' not found')


def flatten_dict(dict_obj):
    result = {}
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            flattened = flatten_dict(value)
            result.update({
                "{}/{}".format(key, kk): fv
                for kk, fv in flattened.items()})
        else:
            result[key] = value
    return result


def _get_palette_map(n_classes):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    color_to_label = {}
    for k in range(0, n_classes):
        red = green = blue = 0
        cls = k
        for j in range(8):
            red = red | (bitget(cls, 0) << 7 - j)
            green = green | (bitget(cls, 1) << 7 - j)
            blue = blue | (bitget(cls, 2) << 7 - j)
            cls = cls >> 3
        color_to_label[(red, green, blue)] = k
    return color_to_label


class ColorPalette:
    def __init__(self, colors):
        if isinstance(colors, int):
            self.color_to_label = _get_palette_map(colors)
        else:
            self.color_to_label = {v: k for k, v in enumerate(colors)}
        self.label_to_color = {v: k for k, v in self.color_to_label.items()}

    def encode_color(self, label_mask):
        """
        Encodes a label mask with its RGB color representation
        Arguments:
            label_mask: A numpy array with dimensions either
                (height, width), (height, width, 1), (batch_size, height, width)
                or (batch_size, height, width, 1)

        Returns: Color encoded representation
        """
        input_shape = label_mask.shape
        if len(input_shape) == 2:
            label_mask = np.reshape(label_mask, [1, *input_shape, 1])
            output_shape = input_shape + (3,)
        elif len(input_shape) == 3 and input_shape[-1] == 1:
            label_mask = np.expand_dims(label_mask, axis=0)
            output_shape = input_shape[:2] + (3,)
        elif len(input_shape) == 3:
            label_mask = np.expand_dims(label_mask, axis=-1)
            output_shape = input_shape + (3,)
        elif len(input_shape) == 4:
            output_shape = input_shape[:3] + (3,)

        palette_mask = np.zeros(label_mask.shape[:3] + (3,), dtype=np.uint8)
        for i, mask in self.label_to_color.items():
            palette_mask[np.where(np.all(label_mask == i, axis=-1))[:3]] = mask

        palette_mask = palette_mask.reshape(output_shape)

        return palette_mask


    def decode_color(self, palette_mask):
        label_mask = np.zeros(palette_mask.shape[:2], dtype=np.uint8)
        for mask, i in self.color_to_label.items():
            label_mask[np.where(np.all(palette_mask == mask, axis=-1))[:2]] = i
        return label_mask


def load_model(
        checkpoint_path, model_cls, transformer_cls,
        model_args={}, transformer_args={}, device=None):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=lambda storage, location: storage)
    model = model_cls(**{**checkpoint.get('model_args', {}), **model_args})
    if device is not None:
        model = model.to(device)
    transformer = transformer_cls(
        'test',
        **{**checkpoint.get('transformer_args', {}), **transformer_args}
    )
    checkpoint['model'] = {
        re.sub('^module[.]', '', key): value
        for key, value in checkpoint['model'].items()
    }
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    def predict(input_tensor):
        with torch.no_grad():
            transformed_image = transformer(
                input_tensor,
                torch.zeros(input_tensor.shape[1:], dtype=torch.long)
            )[0].unsqueeze(0)
            if device is not None:
                transformed_image = transformed_image.to(device)
            return model(transformed_image)
    return predict
