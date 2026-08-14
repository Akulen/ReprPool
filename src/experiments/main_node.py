import time
import argparse
import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx
from typing import cast
from jaxtyping import Array, Float
from tqdm import tqdm

from experiments.utils import (
    complete_model_args, load_run, save_run, make_scheduler, make_optimizer
)

from ggnn.data.node import NodeDataset
from ggnn.node_training import NodeTrainer
from ggnn.models import get_model

CSI = "\x1b[" #]

def run(
    args: argparse.Namespace,
    data: NodeDataset,
    verbose: int=0,
    n_gpus: int=-1,
    force: bool=False
):
    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    dataset_model_params = data.model_params()
    model, state = eqx.nn.make_with_state(
        get_model(args.model, data.task) #type: ignore
    )(
        key=subkey,
        **dataset_model_params,
        **args.model_params
    )
    params, static = eqx.partition(model, eqx.is_array)
    args.param_count = sum(
        x.size
        for x in jax.tree_util.tree_leaves(params)
    )
    if verbose > 5:
        print(params)
    args = complete_model_args(args, dataset_model_params, data.task)
    args.n_samples = {
        tp: data.masks[tp].sum().item()
        for tp in data.masks
    }
    del vars(args)['batch_size']
    del vars(args)['vt_batch_size']
    if verbose > 0:
        print(args)

    if not force:
        results = load_run(args)
        if results is not None:
            return results

    transform, transform_state, scheduler = make_scheduler(args, model, data)
    tx, opt_state = make_optimizer(args, model, scheduler)

    trainer = NodeTrainer(tx, loss=args.loss)

    (
        params, static, state, opt_state, key
    ) = trainer.global_shard(
        params, static, state, opt_state, key
    )
    data.shard_data(trainer)
    data.set_node_step_function(static, trainer.train_step, 'train')
    data.set_node_step_function(static, trainer.eval_step, 'eval')

    data.node_precompile(
        params, state, opt_state, key, 'train', verbose=verbose
    )
    data.node_precompile(
        params, state, opt_state, key, 'eval', verbose=verbose
    )

    best_state = (params, state, opt_state)
    best_val_loss = jnp.inf
    best_epoch = 0
    best_test_losses = None

    train_losses = []
    val_losses = []
    test_losses = []
    lr_history = [(0, opt_state.hyperparams['learning_rate'].item())]
    # dataset_loss = 'dataset' in args.loss
    epoch_time = []

    st = time.time()
    with tqdm(range(1, 1+args.epochs)) as pbar:
      def upd_pbar(train_loss, val_loss, test_loss, lr):
        pbar.set_description(
            f"Loss: {train_loss:.6f} "
            f"[Val: {val_loss}, Test: {test_loss}, lr: {lr:.6f}]"
        )

      for i_epoch in pbar:
        params, state, opt_state, key, losses = data.step(
            params, state, opt_state, key,
            lr_scale=cast(Float[Array, ""], transform_state.scale),
            name='train'
        )
        train_loss = np.array(losses)[0]
        train_losses.append((i_epoch, train_loss))

        _, _, _, key, losses = data.step(
            params, state, opt_state, key,
            lr_scale=cast(Float[Array, ""], transform_state.scale),
            name='eval'
        )
        val_loss, test_loss = np.array(losses)[1:]
        val_losses.append((i_epoch, val_loss))
        test_losses.append((i_epoch, test_loss))

        current_lr_jax = (
            opt_state.hyperparams['learning_rate']
            * cast(Float[Array, ""], transform_state.scale) #type: ignore
        ).item()
        lr_history.append((i_epoch, current_lr_jax))

        upd_pbar(
            train_loss[0].item(), val_loss[0].item(), test_loss[0].item(),
            current_lr_jax
        )

        if transform is not None:
            previous = transform_state.scale #type: ignore
            _, transform_state = transform.update(
                updates=params, #type: ignore
                state=transform_state, value=val_loss[0]
            )
            if previous != transform_state.scale: #type: ignore
                (params, state, opt_state) = best_state

        if best_val_loss > val_loss[0]:
            best_val_loss = val_loss[0]
            best_state = (params, state, opt_state)
            best_epoch = i_epoch
            best_test_losses = test_loss

        epoch_time.append(time.time() - st)
        st = time.time()

        if (
            jnp.isnan(train_loss[0]) or
            lr_history[-1][1] < float(args.scheduler_params.get('min_lr', 0))
        ):
            break

    with np.printoptions(precision=3, suppress=True):
        print(f'Epoch {best_epoch:03}:')
        print(f'Train Losses: {train_losses[best_epoch-1][1]}')
        print(f'Valid Losses: {val_losses[best_epoch-1][1]}')
        if best_test_losses is not None:
            print(f'Test Losses: {best_test_losses}')

    result = {
        'config': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'lr': lr_history,
        'epoch_time': epoch_time,
        'n_gpus': n_gpus,
        'best_val_loss': val_losses[best_epoch-1],
        'best_test_loss': test_losses[best_epoch-1],
    }

    save_run(args, best_state[:2], result)

    return result

def node_exp(
    exp_args: argparse.Namespace,
    data: NodeDataset,
):
    results = []
    for args in exp_args.runs:
        results.append(run(
            args, data, verbose=exp_args.verbose, n_gpus=exp_args.n_gpus,
            force=exp_args.force
        ))
    return results
