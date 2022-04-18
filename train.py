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
from datetime import datetime
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _train(log_dir: str, dataset_path: str, hp: hyper.HyperParameters):
    now = datetime.now().strftime('%Y%m%d_%a_%Hh%Mm%Ss')
    log_dir = os.path.join(log_dir, now)
    os.makedirs(log_dir, exist_ok=True)

    hp_str = yaml.dump(hp)

    train_hp_path = os.path.join(log_dir, 'hp.yaml')
    with open(train_hp_path, mode='w', encoding='utf-8') as f:
        f.write(hp_str)

    train_tag = get_train_tag(hp)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(hp)
    model = model.to(device, non_blocking=True)

    optimizer = get_optimizer(model.parameters(), hp)

    pad_fn = functools.partial(pad_chunk, target_size=32)
    dtaldrs = create_dtaldrs(dataset_path, hp, pad_fn=pad_fn)
    # temporay
    dtaldrs['val'] = dtaldrs['test']

    acc_train, acc_val, acc_test = 0, 0, 0
    with tqdm(total=hp['n_epochs'], initial=0, unit='epoch') as pbar:
        while pbar.n < pbar.total:
            _clear_terminal()
            print(f'Last validation accuracy = {acc_val:.2%}')
            pbar.set_description(f'* Epoch {pbar.n + 1:3d}')

            model.train()
            acc_train = 0
            train_pbar = tqdm(dtaldrs['train'], desc='* Training', leave=False)
            for x, y, _ in train_pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = model(x)  # pylint: disable=not-callable
                optimizer.zero_grad()
                loss = F.nll_loss(F.log_softmax(y_pred, dim=1), y)
                loss.backward()
                optimizer.step()

                acc_train += (y_pred.argmax(dim=1).eq(y).sum().item() /
                              len(dtaldrs['train'].dataset))

                train_pbar.set_description('* Training')

            model.eval()
            acc_val = 0
            for x, y, _ in tqdm(dtaldrs['val'],
                                desc='* Evaluating',
                                leave=False):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                y_pred = model(x)  # pylint: disable=not-callable
                acc_val += (y_pred.argmax(dim=1).eq(y).sum().item() /
                            len(dtaldrs['val'].dataset))

            pbar.update()


def _main(
    log_dir: str,
    dataset_path: str,
    hyper_path: str,
    **_,
):
    if not os.path.exists(dataset_path):
        print(f'A dataset directory "{dataset_path}" does not exist.',
              file=sys.stderr)
        sys.exit(0)

    if not os.path.exists(hyper_path):
        print(f'A hyper parameter settings file "{hyper_path}" does not exist.',
              file=sys.stderr)
        sys.exit(0)

    if not os.path.exists(os.path.join(_CACHE_DIR, _PRT_MODEL)):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with open(os.path.join(_CACHE_DIR, _PRT_MODEL), 'wb') as f:
            utils.request_url_content(_PRT_MODEL_URL, f)

    hp = hyper.load(hyper_path)

    os.makedirs(log_dir, exist_ok=True)

    mp_context = mp.get_context('spawn')
    training_process = mp_context.Process(target=_train,
                                          args=(log_dir, dataset_path, hp))
    training_process.start()
    training_process.join()


if __name__ == '__main__':
    # yapf: disable
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--log_dir', '-l', dest='log_dir', type=str, default='log')
    arg_parser.add_argument('--dataset', '-d', dest='dataset_path', type=str, required=True)
    arg_parser.add_argument('--hyper', dest='hyper_path', type=str, required=True)
    # yapf: enable

    args = vars(arg_parser.parse_args())
    _main(**args)
