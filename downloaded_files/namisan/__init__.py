import os
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from module.san import SANClassifier
from module.pooler import Pooler


TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()

class MTDNNTask:
    def __init__(self, task_def):
        self._task_def = task_def

    def input_parse_label(self, label: str):
        raise NotImplementedError()

    @staticmethod
    def input_is_valid_sample(sample, max_len):
         return len(sample['token_id']) <= max_len 
        
    @staticmethod
    def train_prepare_label(batch, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def prepare_input(batch):
        return {"batch": batch}

    @staticmethod
    def train_prepare_soft_label(softlabels):
        raise NotImplementedError()

    @staticmethod
    def train_build_task_layer(hidden_size, task_def=None, opt=None, prefix="answer"):
        if task_def.enable_san:
            proj = SANClassifier(hidden_size, hidden_size, task_def.n_class, opt, prefix, dropout=task_def.get("dropout_p", 0.0))
        else:
            proj = nn.Linear(hidden_size, task_def.n_class)
        task_layer = nn.Sequential(
            Pooler(
                hidden_size, dropout_p=task_def.dropout_p, actf=task_def.actf
            ),
            proj
        )
        return task_layer
    
    # TODO redesign hypers
    @staticmethod
    def train_forward(sequence_output, premise_mask, hyp_mask, task_layer=None, enable_san=False):
        if enable_san:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:, :max_query, :]
            logits = task_layer(sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            logits = task_layer(sequence_output)
        return logits
    
    @staticmethod
    def test_prepare_label(batch_info, batch):
        batch_info['label'] = [sample["label"] if "label" in sample else None for sample in batch]
    
    @staticmethod
    def test_predict(score, batch_meta, tokenizer=None):
        raise NotImplementedError()

def register_task(name):
    """
        @register_task('Classification')
        class ClassificationTask(MTDNNTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~MTDNNTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, MTDNNTask):
            raise ValueError('Task ({}: {}) must extend MTDNNTask'.format(name, cls.__name__))
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls

def get_task_obj(task_def):
    task_name = task_def.task_type.name
    task_cls = TASK_REGISTRY.get(task_name, None)
    if task_cls is None:
        return None
    
    return task_cls(task_def)

for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("tasks." + file_name)