import os

from experiments.config import parse_exp

exp_args = None
if __name__ == '__main__':
    exp_args = parse_exp()

    from utils.gpu_management import set_gpus
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(exp_args.n_gpus, forcing=False, max_tasks=1)

import time
import argparse
import pickle
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

from ggnn.data import make_dataset, Dataset
from ggnn.data.types import Graph
from ggnn.training import Trainer
from ggnn.models import get_model

jax.config.update("jax_debug_nans", True)

CSI = "\x1b[" #]

def run(
    args: argparse.Namespace,
    data_train: Dataset,
    data_valid: Dataset,
    data_test: Dataset,
    verbose: int=0,
    n_gpus: int=-1,
    force: bool=False
):
    key, subkey = jax.random.split(jax.random.PRNGKey(args.seed))
    dataset_model_params = data_train.model_params()
    model, state = eqx.nn.make_with_state(
        get_model(args.model, data_train.task) #type: ignore
    )(
        key=subkey,
        # n_class=data_train.n_class,
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
    args = complete_model_args(args, dataset_model_params, data_train.task)
    args.n_samples = {
        'train': data_train.n_samples,
        'valid': data_valid.n_samples,
        'test': data_test.n_samples
    }
    if verbose > 0:
        print(args)

    if not force:
        results = load_run(args)
        if results is not None:
            return results

    transform, transform_state, scheduler = make_scheduler(
        args, model, data_train
    )
    tx, opt_state = make_optimizer(args, model, scheduler)

    trainer = Trainer(tx, loss=args.loss)

    (
        params, static, state, opt_state, key
    ) = trainer.global_shard(
        params, static, state, opt_state, key
    )
    data_train.shard_data(trainer)
    data_valid.shard_data(trainer)
    data_test.shard_data(trainer)
    data_train.set_step_function(static, trainer.train_step)
    data_valid.set_step_function(static, trainer.eval_step)
    data_test.set_step_function(static, trainer.eval_step)

    data_train.precompile(params, state, opt_state, key, args.batch_size)
    vt_batch_size = args.vt_batch_size
    if vt_batch_size == 0:
        vt_batch_size = args.batch_size
    data_valid.precompile(params, state, opt_state, key, vt_batch_size)
    data_test.precompile(params, state, opt_state, key, vt_batch_size)

    best_state = (params, state, opt_state)
    best_val_loss = jnp.inf
    best_epoch = 0
    best_test_losses = None

    train_losses = []
    val_losses = []
    test_losses = []
    last_val_loss = 'N/A'
    last_test_loss = 'N/A'
    last_test_losses = None
    lr_history = [(0, opt_state.hyperparams['learning_rate'].item())]
    dataset_loss = 'dataset' in args.loss
    epoch_time = []

    st = time.time()
    with tqdm(range(1, 1+args.epochs)) as pbar:
      def upd_pbar(train_loss, val_loss, test_loss, lr):
        pbar.set_description(
            f"Loss: {train_loss:.6f} "
            f"[Val: {val_loss}, Test: {test_loss}, lr: {lr:.6f}]"
        )

      for i_epoch in pbar:
        (params, state, opt_state, key), losses = data_train.run_batches(
            params, state, opt_state, key, args.batch_size, dataset_loss,
            lr_scale=cast(Float[Array, ""], transform_state.scale)
        )
        train_losses.append((i_epoch, np.array(losses)))
        last_train_loss = losses[0].mean()

        current_lr_jax = (
            opt_state.hyperparams['learning_rate']
            * cast(Float[Array, ""], transform_state.scale) #type: ignore
        ).item()

        upd_pbar(
            last_train_loss, last_val_loss, last_test_loss,
            current_lr_jax
        )

        (params, state, opt_state, key), losses = data_valid.run_batches(
            params, state, opt_state, key, vt_batch_size,
            custom_loss=dataset_loss,
        )
        val_losses.append((i_epoch, np.array(losses)))
        loss_mean = losses[0].mean()
        loss_mean_var = loss_mean
        last_val_loss = f'{loss_mean:.6f}'

        upd_pbar(
            last_train_loss, last_val_loss, last_test_loss,
            current_lr_jax
        )
        if transform is not None:
            previous = transform_state.scale #type: ignore
            _, transform_state = transform.update(
                updates=params, #type: ignore
                state=transform_state, value=loss_mean
            )
            if previous != transform_state.scale: #type: ignore
                (params, state, opt_state) = best_state

        lr_history.append(
            (i_epoch, (
                opt_state.hyperparams['learning_rate']
                * cast(Float[Array, ""], transform_state.scale)
            ).item())
        )

        if i_epoch != 1 and (args.epochs-i_epoch) % args.test_frequency != 0:
            pass
        else:
            (params, state, opt_state, key), losses = data_test.run_batches(
                params, state, opt_state, key, vt_batch_size,
                custom_loss=dataset_loss,
            )
            test_losses.append((i_epoch, np.array(losses)))
            loss_mean = losses[0].mean()
            last_test_loss = f'{loss_mean:.6f}'
            last_test_losses = np.array(losses)
            upd_pbar(
                last_train_loss, last_val_loss, last_test_loss,
                (
                    opt_state.hyperparams['learning_rate']
                    * cast(Float[Array, ""], transform_state.scale)
                ).item()
            )

        if loss_mean_var <= best_val_loss:
            best_val_loss = loss_mean_var
            best_state = (params, state, opt_state)
            best_epoch = i_epoch
            best_test_losses = last_test_losses

        epoch_time.append(time.time() - st)
        st = time.time()

        if (
            jnp.isnan(last_train_loss) or
            lr_history[-1][1] < float(args.scheduler_params.get('min_lr', 0))
        ):
            break

    with np.printoptions(precision=3, suppress=True):
        print(f'Epoch {best_epoch:03}:')
        print(f'Train Losses: {train_losses[best_epoch-1][1].mean(axis=1)}')
        print(f'Valid Losses: {val_losses[best_epoch-1][1].mean(axis=1)}')
        if best_test_losses is not None:
            print(f'Test Losses: {best_test_losses.mean(axis=1)}')

    result = {
        'config': args,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_losses': test_losses,
        'lr': lr_history,
        'epoch_time': epoch_time,
        'n_gpus': n_gpus
    }
    save_run(args, best_state[:2], result)

    return result

import networkx as nx
def jax_to_nx(g: Graph):
    G = nx.DiGraph()
    for i in range(g.n_nodes):
        G.add_node(i)
    for u, v in g.edges[:g.n_edges]:
        G.add_edge(u.item(), v.item())
    return G

def graph_exp(
    exp_args: argparse.Namespace,
    data_train: Dataset,
    data_valid: Dataset,
    data_test: Dataset,
):
    results = []
    for args in exp_args.runs:
        results.append(run(
            args, data_train, data_valid, data_test, verbose=exp_args.verbose,
            n_gpus=exp_args.n_gpus, force=exp_args.force
        ))
    return results


if exp_args is not None:
    data_train = make_dataset(
        exp_args.dataset,
        cache=not exp_args.debug,
        # category='train',
        shuffle=True,
        verbose=exp_args.verbose>1
    )
    if data_train.task == 'node':
        from experiments.main_node import node_exp
        from ggnn.data.node import NodeDataset
        assert isinstance(data_train, NodeDataset)
        results = node_exp(exp_args, data_train)
    else:
        assert data_train.task == 'graph'
        data_valid = make_dataset(
            exp_args.dataset,
            cache=not exp_args.debug,
            category='valid',
            shuffle=False,
            verbose=exp_args.verbose>1
        )
        data_test = make_dataset(
            exp_args.dataset,
            cache=not exp_args.debug,
            category='test',
            shuffle=False,
            verbose=exp_args.verbose>1
        )

        mods = {}
        if 'modularity' in exp_args.loss:
            for name, dt in [
                ('train', data_train), ('valid', data_valid), ('test', data_test)
            ]:
                continue
                cur_mod = 0
                for i in range(dt.samples.data.n_nodes.shape[0]): # type: ignore
                    g = cast(Graph, dt.samples.data[i])
                    g = jax_to_nx(g)
                    clusters = nx.algorithms.community.louvain_communities(g)
                    cur_mod += nx.algorithms.community.modularity(g, clusters)
                mods[name] = cur_mod / dt.samples.data.n_nodes.shape[0] # type: ignore
            print(mods)
            # {'train': 0.6477776488954075, 'valid': 0.6481762183182765, 'test': 0.6452429486200214}

        results = graph_exp(exp_args, data_train, data_valid, data_test)

    os.makedirs(f'experiments/results/{exp_args.name}', exist_ok=True)
    with open(f'experiments/results/{exp_args.name}/results.pkl', 'wb') as f:
        pickle.dump((exp_args, results), f)
