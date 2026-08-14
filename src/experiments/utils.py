import argparse
from collections import namedtuple
import hashlib
import json
import os
import pickle
from typing import cast, Optional, Tuple
import optax
import equinox as eqx

from ggnn.data import Dataset
from ggnn.data.types import Graph
from ggnn.models import get_model_params

CSI = "\x1b[" #]

def complete_model_args(
    args: argparse.Namespace, dataset_model_params: dict, task: str='graph'
) -> argparse.Namespace:
    for param, sign in get_model_params(args.model, task).items():
        if param in ['self', 'key', 'n_class']: continue
        if param not in args.model_params:
            if param in dataset_model_params:
                args.model_params[param] = dataset_model_params[param]
            else:
                args.model_params[param] = sign.default
    return args

def get_hash(args: argparse.Namespace) -> str:
    return hashlib.sha256(
        json.dumps(vars(args), sort_keys=True).encode('utf-8')
    ).hexdigest()

def load_run(
    args: argparse.Namespace, path: str='experiments/cache'
) -> Optional[dict]:
    hs = get_hash(args)
    path = f'{path}/{hs}/results.pkl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            results = pickle.load(f)
            if vars(args) != vars(results['config']):
                raise ValueError(
                    "Cached results do not match args.\n"
                    f"{CSI}31m{args}{CSI}0m\n"
                    f"{CSI}34m{results['config']}{CSI}0m"
                )
        print(f'{CSI}34mUsing cached results for{CSI}0m {hs}')
        return results
    return None

def save_run(
    args: argparse.Namespace, ckpt: Tuple, result, path: str='experiments/cache'
):
    hs = get_hash(args)
    path = f'{path}/{hs}'

    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(
        f'{path}/model-{args.model}.eqx',
        ckpt
    )
    with open(f'{path}/results.pkl', 'wb') as f:
        pickle.dump(result, f)

def make_scheduler(args: argparse.Namespace, model: eqx.Module, data: Dataset):
    transform = None
    transform_state = namedtuple('TransformState', ['scale'])(1.0)

    if args.scheduler == 'warmup_cosine':
        raise NotImplementedError
    elif args.scheduler == 'constant':
        scheduler = args.lr
    elif args.scheduler == 'reducelr':
        transform = optax.contrib.reduce_on_plateau(
            factor=args.scheduler_params.get('factor', 0.1),
            patience=args.scheduler_params.get('patience', 10) + 1,
            rtol=args.scheduler_params.get('rtol', 1e-4),
            atol=args.scheduler_params.get('atol', 0),
            cooldown=args.scheduler_params.get('cooldown', 0),
        )
        transform_state = transform.init(
            eqx.filter(model, eqx.is_inexact_array)
        )
        assert isinstance(transform_state, optax.contrib.ReduceLROnPlateauState)
        scheduler = args.lr
    elif args.scheduler == 'periodscale':
        match data.task:
            case 'graph':
                steps_per_epoch = cast(
                    Graph, data.make_batches(args.batch_size).data
                ).n_nodes.shape[0]
            case 'node':
                steps_per_epoch = 1
            case _:
                raise ValueError(f'Unknown task {data.task}')

        period = args.scheduler_params.get('period', 10)
        scheduler = optax.piecewise_constant_schedule(
            init_value=args.lr,
            boundaries_and_scales={
                epoch*steps_per_epoch: args.scheduler_params.get('factor', 0.5)
                for epoch in range(period, args.epochs, period)
            }
        )
    else:
        raise ValueError(f'Unknown scheduler {args.scheduler}')
    
    return transform, transform_state, scheduler

def make_optimizer(args: argparse.Namespace, model: eqx.Module, scheduler):
    if args.optimizer == 'lion':
        tx = optax.lion
    elif args.optimizer == 'adam':
        tx = optax.adam
    elif args.optimizer == 'adamw':
        tx = optax.adamw
    else:
        raise NotImplementedError(f'Unknown optimizer {args.optimizer}')
    if args.optimizer == 'adamw':
        tx = optax.inject_hyperparams(tx)(
            learning_rate=scheduler,
            weight_decay=args.weight_decay
        )
    else:
        tx = optax.inject_hyperparams(tx)(
            learning_rate=scheduler,
        )
    opt_state = tx.init(eqx.filter(model, eqx.is_inexact_array))

    return tx, opt_state
