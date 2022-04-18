from __future__ import absolute_import

import argparse
import functools
import hashlib
import itertools
import multiprocessing as mp
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as tv
import yaml
from tqdm import tqdm

import data
import gnuoct
import hyper
import utils
from data import augmentations as augs

_CACHE_DIR = './.cache'
_PRT_MODEL = 'R50x1_224.npz'
_PRT_MODEL_URL = 'https://storage.googleapis.com/bit_models/distill/R50x1_224.npz'


def _clear_terminal():
    os.system('clear' if not os.name == 'nt' else 'cls')


def get_train_tag(hp: Union[str, hyper.HyperParameters]) -> str:
    hp = hp if not isinstance(hp, (dict, defaultdict)) else yaml.dump(hp)
    sha256 = hashlib.sha256()
    sha256.update(hp.encode(encoding='utf-8'))
    return sha256.hexdigest()


def get_model(hp: hyper.HyperParameters) -> nn.Module:
    if hp['model'] == 'OCTVolumeConv1dNet':
        raise ValueError('Not implemented.')

    if hp['model'] == 'OCTVolumeCBAMNet':
        raise ValueError('Not implemented.')

    if hp['model'] == 'OCTVolumeFCAttnNet':
        raise ValueError('Not implemented.')

    if hp['model'] == 'OCTVolumeTokenAttnNet':
        prt_model_path = os.path.join(_CACHE_DIR, _PRT_MODEL)
        model = gnuoct.models.OCTVolumeTokenAttnNet(prt_model_path)
        return model

    raise ValueError('Unrecognized model type: {}.'.format(hp['model']))


def get_optimizer(params: Any,
                  hp: hyper.HyperParameters) -> torch.optim.Optimizer:
    if hp['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params,
            lr=hp['optimizer.base_lr'],
            momentum=hp['SGD.momentum'],
            weight_decay=hp['optimizer.weight_decay'],
        )
        return optimizer
    if hp['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            params,
            lr=hp['optimizer.base_lr'],
            weight_decay=hp['optimizer.weight_decay'],
        )
        return optimizer

    raise ValueError('Unrecognized optimizer type: {}.'.format(hp['optimizer']))


def get_augmentation_list() -> Tuple[Tuple[augs.AugmentationOp, float, float]]:
    return (
        (augs.AutoContrast(), 0, 1),
        (augs.Equalize(), 0, 1),
        (augs.Invert(), 0, 1),
        (augs.Rotate(), 0, 30),
        # (augs.Posterize(), 0, 4),
        (augs.Solarize(), 0, 256),
        (augs.SolarizeAdd(), 0, 110),
        (augs.Color(), 0.1, 1.9),
        (augs.Contrast(), 0.4, 1.6),
        (augs.Brightness(), 0.4, 1.6),
        (augs.Sharpness(), 0.1, 1.9),
        (augs.ShearX(), 0., 0.3),
        (augs.ShearY(), 0., 0.3),
        (augs.TranslateXAbs(), 0., 10),
        (augs.TranslateYAbs(), 0., 10),
        (augs.Identity(), 0, 1),
    )


def get_transform(
    dataset_type: Union[Literal['train'], Literal['val'], Literal['test']],
    hp: Optional[hyper.HyperParameters] = None,
) -> torch.utils.data.DataLoader:
    BICUBIC = tv.transforms.InterpolationMode.BICUBIC  # pylint: disable=invalid-name

    if dataset_type == 'train':
        if hp is not None:
            augmentation_list = get_augmentation_list()
            # pylint: disable=line-too-long
            # yapf: disable
            return tv.transforms.Compose([
                data.RandAugment(n=hp['aug.rand_augment.N'],
                                 m=hp['aug.rand_augment.M'],
                                 augmentation_list=augmentation_list),
                tv.transforms.ColorJitter(saturation=0.4, hue=0.4),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomResizedCrop(224, scale=(0.9, 1.1), interpolation=BICUBIC),
                tv.transforms.ToTensor(),
            ])
            # yapf: enable

        warnings.warn(('Transform for trainset was requested, '
                       'but hyper parameters were not provided. '
                       'It\'ll return default transform for validation.'),
                      category=UserWarning)

    if dataset_type in ('train', 'val', 'test'):
        return tv.transforms.Compose([
            tv.transforms.Resize(224, interpolation=BICUBIC),
            tv.transforms.ToTensor(),
        ])

    raise ValueError(f'"{dataset_type}" is not valid dataset type.')


def pad_chunk(
    chunk: List[torch.Tensor],
    target_size: int,
) -> List[torch.Tensor]:
    shortage = target_size - len(chunk)
    pad = chunk[0].new_zeros(chunk[0].size())
    chunk = ([pad.clone() for _ in range(shortage // 2)] + chunk +
             [pad.clone() for _ in range(shortage // 2 + shortage % 2)])
    del pad
    return chunk


def create_datasets(
    dataset_dir: str,
    hp: Optional[hyper.HyperParameters] = None,
) -> Dict[Union[Literal['train'], Literal['test'], Literal['val']],
          gnuoct.GNUOCTVolume]:
    data_ = gnuoct.harvest(dataset_dir, return_relative_path=False)
    datasets = {}
    for dtype in ('train', 'test', 'val'):
        if not (f'x_{dtype}' in data_ and f'y_{dtype}' in data_):
            continue

        x, y = data_[f'x_{dtype}'], data_[f'y_{dtype}']
        transform = get_transform(dtype, hp=hp)
        datasets[dtype] = gnuoct.GNUOCTVolume(zip(x, y), transform=transform)

    return datasets


@torch.no_grad()
def _collate_fn(
    batch: Tuple[Tuple[torch.Tensor], int],
    pad_fn: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    chunks, labels = zip(*batch)
    if pad_fn is not None:
        chunks = [pad_fn(chunk) for chunk in chunks]
    chunk_sizes = tuple(len(chunk) for chunk in chunks)
    with torch.no_grad():
        chunks = torch.stack(tuple(itertools.chain(*chunks)))
    return (chunks, torch.LongTensor(labels), chunk_sizes)


def create_dtaldrs(
    dataset_dir: str,
    hp: hyper.HyperParameters,
    pad_fn: Optional[Callable[[List[torch.Tensor]], torch.Tensor]] = None,
) -> Dict[Union[Literal['train'], Literal['test'], Literal['val']],
          torch.utils.data.DataLoader]:
    dtaldrs = {}
    for dtype, dataset in create_datasets(dataset_dir, hp).items():
        dtaldrs[dtype] = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=hp['sz_batch'],
            shuffle=(dtype == 'train'),
            collate_fn=functools.partial(_collate_fn, pad_fn=pad_fn),
            num_workers=mp.cpu_count(),
            persistent_workers=True,
            pin_memory=True,
        )

    return dtaldrs


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_sizes: Tuple[int],
    lamb: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """https://github.com/facebookresearch/mixup-cifar10"""

    if any(chunk_size != chunk_sizes[0] for chunk_size in chunk_sizes):
        raise RuntimeError(
            'All of OCT Volume size must be same to perform mixup.')

    indices = np.random.permutation(len(chunk_sizes))
    x = x.view(len(chunk_sizes), chunk_sizes[0], *x.shape[-3:])
    x = lamb * x + (1 - lamb) * x[indices]
    return x.view(-1, *x.shape[-3:]), y, y[indices]


def mixup_criterion(
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y_pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lamb: float,
) -> torch.Tensor:
    """https://github.com/facebookresearch/mixup-cifar10"""

    return lamb * criterion(y_pred, y_a) + (1 - lamb) * criterion(y_pred, y_b)


def train_once(
    train_dtaldr: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    hp: hyper.HyperParameters,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[float, float]:
    model.train()

    if device is None:
        device = next(model.parameters()).device

    total = 0
    total_eq = 0
    total_loss = 0.
    train_pbar = tqdm(train_dtaldr, leave=False)
    for i, (x, y, chunk_sizes) in enumerate(train_pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if hp['aug.mixup.use']:
            alpha = hp['aug.mixup.alpha']
            lamb = np.random.beta(alpha, alpha)
            x, y_a, y_b = mixup_data(x, y, chunk_sizes, lamb)

        y_pred = model(x)
        if hp['aug.mixup.use']:
            loss = mixup_criterion(criterion, y_pred, y_a, y_b, lamb)
        else:
            loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += len(chunk_sizes)
        total_loss += loss.item()
        with torch.no_grad():
            if hp['aug.mixup.use']:
                y_a, y_b = (y_b, y_a) if lamb < 0.5 else (y_a, y_b)
                _, predictions = y_pred.topk(k=2, dim=-1)
                top1, top2 = predictions[:, 0], predictions[:, 1]
                total_eq += ((1 - lamb) * torch.eq(y_a, top1).sum().item() +
                             lamb * torch.eq(y_b, top2).sum().item())
            else:
                top1 = torch.argmax(y_pred, -1)
                total_eq += torch.eq(y, top1).sum().item()

        desc = f'> Training | ACC={total_eq / total:.2%} | LOSS={total_loss / i:.6f}'
        train_pbar.set_description(desc)

    return total_eq / total, total_loss / len(train_dtaldr)


@torch.no_grad()
def evaluate(
    eval_dtaldr: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[float, float]:
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    total = 0
    total_eq = 0
    total_loss = 0.
    eval_pbar = tqdm(eval_dtaldr, leave=False)
    for i, (x, y, chunk_sizes) in enumerate(eval_pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_pred = model(x)
        loss = criterion(y_pred, y)
        total += len(chunk_sizes)
        total_eq += torch.eq(y_pred.argmax(-1), y).sum().item()
        total_loss += loss.item()

        desc = f'> Evaluating | ACC={total_eq / total:.2%} | LOSS={total_loss / i:.6f}'
        eval_pbar.set_description(desc)

    return total_eq / total, total_loss / len(eval_dtaldr)


def _get_text_graph(values: Tuple[float], to_string: str) -> str:
    if len(values) == 0:
        return '-'

    text_graph = to_string(values[0])
    for i in range(1, len(values)):
        text_graph += '  ↗ ' if values[i - 1] < values[i] else ''
        text_graph += '  ↘ ' if values[i] < values[i - 1] else ''
        text_graph += '  → ' if values[i - 1] == values[i] else ''
        text_graph += to_string(values[i])

    return text_graph.strip()


def _save_ckpt(
    ckpt_path: str,
    model: nn.Module,
    optimzier: torch.optim.Optimizer,
    epoch: int,
    history: Dict[str, List[float]],
):
    ckpt = {
        'srs': random.getstate(),
        'nrs': np.random.get_state(),
        'trs': torch.random.get_rng_state(),
        'model': model.state_dict(),
        'optimizer': optimzier.state_dict(),
        'epoch': epoch,
        'history': history,
    }

    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, '.ckpt')

    torch.save(ckpt, ckpt_path)


def _load_ckpt(
    ckpt_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    sync_random_state: bool = False,
) -> Dict[str, Any]:

    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, '.ckpt')

    ckpt = torch.load(ckpt_path)

    if sync_random_state:
        random.setstate(ckpt['srs'])
        np.random.set_state(ckpt['nrs'])
        torch.random.set_rng_state(ckpt['trs'])

    if model is not None:
        model.load_state_dict(ckpt['model'])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])

    if None not in (model, optimizer):
        device = next(model.parameters()).device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return ckpt


def train(
    log_dir: str,
    dataset_path: str,
    resume: bool,
    hp: hyper.HyperParameters,
):
    train_tag = get_train_tag(hp)
    log_dir = os.path.join(log_dir, train_tag)
    if not resume and os.path.exists(log_dir):
        print(
            'Training that had been performed with same hyperparameters '
            'is detected.',
            file=sys.stderr)
        return

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, '.hp.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml.dump(hp))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(hp)
    model = model.to(device, non_blocking=True)
    if hasattr(model, 'init_weights'):
        model.init_weights()

    optimizer = get_optimizer(model.parameters(), hp)

    pad_fn = functools.partial(pad_chunk, target_size=32)
    dtaldrs = create_dtaldrs(dataset_path, hp, pad_fn=pad_fn)

    label_smoothing = hp.setdefault('label_smoothing', 0.)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    criterion = criterion.to(device)

    epoch, history = 1, defaultdict(list)
    if resume:
        ckpt = _load_ckpt(log_dir, model, optimizer, sync_random_state=True)
        epoch, history = ckpt['epoch'] + 1, ckpt['history']

    # pylint: disable=invalid-name
    purge_step = len(dtaldrs['train']) * epoch
    tb = utils.TrickySummaryWriter(log_dir, purge_step=purge_step)
    for epoch in range(epoch, hp['n_epochs'] + 1):
        _clear_terminal()

        print('EPOCH', f'{epoch}/{hp["n_epochs"]}', sep='\t')
        print('* TRAIN_TAG', train_tag, sep='\t')

        for metric_name in ('acc_train', 'loss_train', 'acc_val', 'loss_val'):

            def _to_string(metric_name: str) -> Callable[[float], str]:
                is_acc = metric_name.startswith('acc')
                vformat = ('{:.6f}', '{:6.4%}')[int(is_acc)]
                return vformat.format

            text_graph = _get_text_graph(history[metric_name][-6:],
                                         _to_string(metric_name))
            print(f'* {metric_name.upper()}', text_graph, sep='\t')

        # yapf: disable
        acc_train, loss_train = train_once(dtaldrs['train'], model, criterion, optimizer, hp)
        # yapf: enable
        acc_val, loss_val = evaluate(dtaldrs['test'], model, criterion)

        history['acc_train'].append(acc_train)
        history['loss_train'].append(loss_train)
        history['acc_val'].append(acc_val)
        history['loss_val'].append(loss_val)

        for metric_name in ('acc_train', 'loss_train', 'acc_val', 'loss_val'):
            tb.add_scalar(metric_name.replace('_', '/'),
                          history[metric_name][-1],
                          len(dtaldrs['train']) * epoch)

        _save_ckpt(log_dir, model, optimizer, epoch, history)
        if max(*history['acc_val']) <= acc_val:
            pass


def _main(
    log_dir: str,
    dataset_path: str,
    hyper_path: str,
    resume: bool,
    **_,
):
    if not os.path.exists(dataset_path):
        print(f'A dataset directory "{dataset_path}" does not exist.',
              file=sys.stderr)
        sys.exit(0)

    if resume and not hyper_path.endswith(('.yaml', '.yml')):
        hyper_path = os.path.join(log_dir, hyper_path, '.hp.yaml')
    if not os.path.exists(hyper_path):
        print(f'A hyperparameter file "{hyper_path}" does not exist.',
              file=sys.stderr)
        sys.exit(0)

    if not os.path.exists(os.path.join(_CACHE_DIR, _PRT_MODEL)):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, _PRT_MODEL), 'wb') as f:
            utils.request_url_content(_PRT_MODEL_URL, f)

    train(log_dir=log_dir,
          dataset_path=dataset_path,
          resume=resume,
          hp=hyper.load(hyper_path))


if __name__ == '__main__':
    # yapf: disable
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_dir', '-l', dest='log_dir', type=str, default='log')
    arg_parser.add_argument('--dataset', '-d', dest='dataset_path', type=str, required=True)
    arg_parser.add_argument('--hyper', dest='hyper_path', type=str, required=True)
    arg_parser.add_argument('--resume', '-r', dest='resume', action='store_true')
    # yapf: enable

    args = vars(arg_parser.parse_args())
    _main(**args)
