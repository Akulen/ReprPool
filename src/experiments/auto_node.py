import os

exp_args = None
if __name__ == '__main__':
    from experiments.config import parse_auto_exp
    exp_args = parse_auto_exp()

    from utils.gpu_management import set_gpus
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(exp_args.n_gpus, forcing=False)

import copy
import argparse
import numpy as np
import optuna

from ggnn.data.node import NodeDataset
from ggnn.data import make_dataset
from experiments.main_node import run


def node_exp_optuna(
    exp_args: argparse.Namespace,
    data: NodeDataset,
    n_trials: int = 50,
    study_name: str = None,
    storage: str = None,
    seed: list = [42],
):
    base_run = exp_args  # no more runs list

    # Build search space from model_params that are lists
    EXCLUDE_FROM_SEARCH = {
        'loss', 'conf', 'seed', 'models', 'model_params',
        'batch_size', 'vt_batch_size', 'n_trials', 'optuna_storage'
    }

    def is_search_space(k, v):
        if k == 'ks': return False
        return (
            isinstance(v, list) and
            len(v) > 0 and
            all(isinstance(x, (int, float, str, bool)) for x in v)
        )

    searchable = {
        k: v for k, v in vars(base_run).items()
        if k not in EXCLUDE_FROM_SEARCH and is_search_space(k, v)
    }
    searchable_model = {
        k: v for k, v in base_run.model_params.items()
        if is_search_space(k, v)
    }
    # Remove searchable keys from model_params so they don't conflict
    for k in searchable_model:
        base_run.model_params[k] = None

    print(f"Searchable top-level params: {list(searchable.keys())}")
    print(f"Searchable model params:     {list(searchable_model.keys())}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial) -> float:
        trial_args = copy.deepcopy(base_run)

        for k, choices in searchable.items():
            setattr(trial_args, k, trial.suggest_categorical(k, choices))

        trial_args.model_params = copy.deepcopy(base_run.model_params)
        for k, choices in searchable_model.items():
            trial_args.model_params[k] = trial.suggest_categorical(k, choices)

        sampled_str = {
            **{k: getattr(trial_args, k) for k in searchable},
            **{k: trial_args.model_params[k] for k in searchable_model}
        }

        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        for _seed in seed:
            trial_args.seed = _seed
            # result = run(
            #     copy.deepcopy(trial_args), data,
            #     verbose=1,
            #     n_gpus=exp_args.n_gpus,
            #     force=True
            # )
            try:
                result = run(
                    copy.deepcopy(trial_args), data,
                    verbose=0,
                    n_gpus=exp_args.n_gpus,
                    force=True
                )
                val_losses.append(result['best_val_loss'][1][0])
                val_accs.append(result['best_val_loss'][1][1])
                test_losses.append(result['best_test_loss'][1][0])
                test_accs.append(result['best_test_loss'][1][1])
            except Exception as e:
                print(f"Trial {trial.number} seed={_seed} failed: {e}")
                return float('inf')

        mean_val = float(np.mean(val_losses))
        mean_val_acc = float(np.mean(val_accs))
        mean_test_loss = float(np.mean(test_losses))
        mean_test_acc = float(np.mean(test_accs))
        print(
            f"Trial {trial.number:3d} | val_loss={mean_val:.4f} | val_acc={mean_val_acc:.4f} | "
            + f" | test_loss={mean_test_loss:.4f} | test_acc={mean_test_acc:.4f} | "
            + " | ".join(f"{k}={v}" for k, v in sampled_str.items())
        )
        return mean_val_acc

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name or exp_args.name,
        storage=storage,
        load_if_exists=True
    )
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Optuna Best Trial ===")
    print(f"  Val loss: {study.best_trial.value:.4f}")
    print(f"  Params:   {study.best_trial.params}")
    return study


if exp_args is not None:
    data = make_dataset(
        exp_args.dataset,
        cache=not exp_args.debug,
        shuffle=True,
        verbose=exp_args.verbose > 1
    )
    assert isinstance(data, NodeDataset), \
        "auto_node.py only supports node classification datasets"

    n_trials = getattr(exp_args, 'n_trials', 50)
    seed = getattr(exp_args, 'seed', [42])
    storage = getattr( exp_args, 'optuna_storage', None)
    if storage is None:
        storage = 'sqlite:////home/trigaux/GGNN/logs/optuna.db'
    print(storage)

    node_exp_optuna(
        exp_args, data,
        n_trials=n_trials,
        seed=seed,
        storage=storage,
    )
