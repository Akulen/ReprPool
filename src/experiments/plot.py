import os
import warnings
import argparse
import copy
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import pickle
from typing import List, Optional, Tuple
import matplotlib as mpl
from experiments.config import parse_plot

mpl.rcParams['agg.path.chunksize'] = 10000

cmap = None

def str_field(field: str, value):
    if field == 'lr':
        return f'{value:.0e}'
    return str(value)

def fields_to_label(params, fields, prefix=''):
    label = ''
    cnt = 0
    start = True
    for field, (value, length) in fields.items():
        if len(value) == 1:
            continue
        if not start:
            label += ','
        start = False
        if prefix+field in params:
            cnt += 1
            label += \
                f'{field}={str_field(field, params[prefix+field]):<{length}}'
        else:
            label += \
                ' ' * (len(field) + 1 + length)
    return label, cnt

def make_labels(
    configs: List[argparse.Namespace],
    show_param_count=True,
    average_runs=False
):
    longest_model_name = 0
    field_lengths = {}
    field_values = {}
    params_list = []
    for config in configs:
        params = {}
        for k, v in vars(config).items():
            if k in ['model', 'loss', 'test_frequency']: continue
            if average_runs and k == 'seed': continue
            if not show_param_count and k == 'param_count': continue
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            params[f'{k}.{k2}.{k3}'] = v3
                    else:
                        params[f'{k}.{k2}'] = v2
            else:
                params[k] = v
        params_list.append(params)
        longest_model_name = max(longest_model_name, len(config.model))
        for k, v in params.items():
            if k == 'model_params.hook_args': continue # TODO
            if k not in field_lengths:
                field_lengths[k] = 0
                field_values[k] = set()
            field_lengths[k] = max(field_lengths[k], len(str_field(k, v)))
            if isinstance(v, list):
                v = tuple(v)
            field_values[k].add(v)

    labels = []
    for config, params in zip(configs, params_list):
        label_model_params, cnt = fields_to_label(
            params,
            {
                field[13:]: (field_values[field], field_lengths[field])
                for field in field_values
                if field.startswith('model_params.')
            },
            'model_params.'
        )
        label_params, _ = fields_to_label(
            params,
            {
                field: (field_values[field], field_lengths[field])
                for field in field_values
                if not field.startswith('model_params.')
            }
        )
        label = f'{config.model:<{longest_model_name}}'
        if cnt > 0:
            label += f'[{label_model_params}]'
        if len(label_params) > 0:
            label += ',' + label_params
        labels.append(label)
    return labels

def smooth(Y, smooth_window):
    extra = (smooth_window + 1) // 2
    Y = np.repeat(Y, [extra] + [1] * (len(Y)-2) + [extra])
    Y = np.convolve(Y, np.ones(smooth_window) / smooth_window, mode='valid')
    return Y

def plot_curve(ax, X, Y, color, args):
    min_index = np.argmin(Y)
    max_index = np.argmax(Y)

    min_x = X[min_index]
    min_y = Y[min_index]
    max_x = X[max_index]
    max_y = Y[max_index]

    ax.plot(
        X,
        Y,
        color=cmap(color),
        **args
    )

    if min_x > 1:
        ax.plot(min_x, min_y, marker='x', color=cmap(color), markersize=5)
    if max_x > 1:
        ax.plot(max_x, max_y, marker='x', color=cmap(color), markersize=5)

def plot_single(
    ax: Axes,
    args: dict,
    color: int,
    loss: int,
    results: List[Tuple[float, List[List[float]]]],
    max_epoch_size: int,
    group_batches: bool=True,
    smooth_window: int = 1,
    x_max: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    X, Y = list(zip(*results))
    Y = [y[loss] for y in Y]
    if not group_batches:
        X = np.concatenate([
            x - 1 + np.arange(len(y)) / max_epoch_size
            for x, y in zip(X, Y)
        ])
    Y = np.concatenate([
        [y.mean()] if group_batches else y
        for y in Y
    ])
    if smooth_window > 1:
        Y = smooth(Y, smooth_window)
    X = np.array(X)
    if x_max is not None:
        X = X[X <= x_max]
        Y = Y[:len(X)]
    plot_curve(ax, X, Y, color, args)
    return Y

def plot_average(
    ax: Axes,
    args: dict,
    color: int,
    loss: int,
    results: List[List[Tuple[float, List[List[float]]]]],
    max_epoch_size: int,
    group_batches: bool = True,
    smooth_window: int = 1,
    ci_alpha: float = 0.1,
    x_max: Optional[int] = None
) -> np.ndarray:
    if len(results) == 0:
        return np.array([])
    Ys = []
    X_ref = None

    for run in results:
        X, Y = list(zip(*run))
        Y = [y[loss] for y in Y]

        if not group_batches:
            X_run = np.concatenate([
                x - 1 + np.arange(len(y)) / max_epoch_size
                for x, y in zip(X, Y)
            ])
        else:
            X_run = np.asarray(X)

        Y_run = np.concatenate([
            [y.mean()] if group_batches else y
            for y in Y
        ])
        if smooth_window > 1:
            Y_run = smooth(Y_run, smooth_window)
        X = np.array(X)

        if X_ref is None or len(X_run) > len(X_ref):
            X_ref = X_run
        else:
            assert np.allclose(X_ref[:len(X_run)], X_run[:len(X_ref)])

        Ys.append(Y_run)
    assert X_ref is not None

    max_len = len(X_ref)
    padded_Ys = []
    for y in Ys:
        if len(y) < max_len:
            curr_y = np.full(max_len, np.nan)
            curr_y[:len(y)] = y
            padded_Ys.append(curr_y)
        else:
            padded_Ys.append(y)

    Y = np.stack(padded_Ys, axis=0)
    if x_max is not None:
        X_ref = X_ref[X_ref <= x_max]
        Y = Y[:,:len(X_ref)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Y_mean = np.nanmean(Y, axis=0)
        Y_std = np.nanstd(Y, axis=0) / 2

    plot_curve(ax, X_ref, Y_mean, color, args)
    ax.fill_between(
        X_ref,
        Y_mean - Y_std,
        Y_mean + Y_std,
        color=cmap(color),
        alpha=ci_alpha,
        linewidth=0
    )

    return Y_mean

def plot(
    exp_args, labels, results, plot_args, path, group_batches=True, loss=None
):
    if loss is None:
        loss = exp_args.loss
    if isinstance(loss, List):
        if plot_args.y_min is not None and len(plot_args.y_min) == 1:
            plot_args.y_min = plot_args.y_min * len(loss)
        if plot_args.y_max is not None and len(plot_args.y_max) == 1:
            plot_args.y_max = plot_args.y_max * len(loss)
        for i, l in enumerate(loss):
            plot(
                exp_args, labels, results, plot_args, path,
                group_batches=group_batches, loss=(i, l)
            )
        return
    i_loss, loss = loss
    fig, ax = plt.subplots(figsize=(plot_args.width, plot_args.height))
    ax_lr = None
    if plot_args.plot_lr:
        ax_lr = ax.twinx()
        ax_lr.set_yscale('log')
    # ax.set_xlabel('Epoch' if group_batches else 'Batch')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(f'Loss ({loss})')
    ax.set_title(f'{exp_args.name}{'' if group_batches else ' Batch'} Losses')

    set_config = {
        'train_losses': {
            'name': 'Train',
            'line': {
            }
        },
        'val_losses': {
            'name': 'Validation',
            'line': {
                'linewidth': 0.75,
            }
        },
        'test_losses': {
            'name': 'Test',
            'line': {
                'linestyle': 'dashdot',
                'linewidth': 1,
            }
        }
    }
    set_ci_alpha = {
        'train_losses':
            0.3 if (
                    not plot_args.focus_valid
                and not plot_args.focus_test
            ) else 0.1,
        'val_losses': 0.3 if plot_args.focus_valid else 0.1,
        'test_losses': 0.3 if plot_args.focus_test else 0.1
    }
    for dataset in set_config:
        ax.plot(
            [], [], **set_config[dataset]['line'], color='grey',
            label=f'{set_config[dataset]["name"]} Loss'
        )
    if ax_lr is not None:
        ax.plot(
            [], [], linestyle=':', linewidth='0.5', color='gray',
            label='Learning Rate'
        )

    # max_epochs = max(run.epochs for run in exp_args.runs)
    max_epochs = 0
    for dataset in set_config:
        max_epochs = max(
            max_epochs,
            max(
                result[dataset][-1][0]+1
                for result in results
                for _, ys in result[dataset]
            )
        )
    test_epochs = exp_args.runs[0].test_frequency
    n_epochs = exp_args.runs[0].epochs
    for run in exp_args.runs[1:]:
        if run.test_frequency != test_epochs:
            test_epochs = None
            break
        if run.epochs != n_epochs:
            test_epochs = None
            break
    for epoch in range(max_epochs+1):
        is_test_epoch = False
        if test_epochs is not None:
            if test_epochs < max_epochs // 20:
                test_epochs = max_epochs // 20
            is_test_epoch = (n_epochs - epoch) % test_epochs == 0
        if is_test_epoch and group_batches:
            ls = "dotted" if plot_args.only_test_indicators else "dashed"
            ax.axvline(
                epoch,
                linestyle=ls,
                color="0.7", lw=1
            )
        if not is_test_epoch and plot_args.only_test_indicators:
            continue
        ls = "dashed" if is_test_epoch else "dotted"
        if plot_args.only_test_indicators:
            ls = "dotted"
        if not group_batches:
            ax.axvline(
                epoch,
                linestyle=ls, color="0.7", lw=1
            )

    ####################
    if ax_lr is not None:
        ax_lr.set_ylabel('Learning Rate')
        if not plot_args.average_runs:
            for i, result in enumerate(results):
                ax_lr.plot(
                    *list(zip(*result['lr'])),
                    linestyle=':', linewidth=0.5, color=cmap(i)
                )
        else:
            _grouped_lrs = {}
            for i, run in enumerate(results):
                seed = run['config'].seed
                run['config'].seed = None
                hs = hashlib.sha256(json.dumps(
                    vars(run['config']),
                    sort_keys=True
                ).encode('utf-8')).hexdigest()
                run['config'].seed = seed
                if hs not in _grouped_lrs:
                    _grouped_lrs[hs] = []
                _grouped_lrs[hs].append((i, run))
            grouped_runs = []
            for i, (hs, runs) in enumerate(_grouped_lrs.items()):
                label = labels[runs[0][0]]
                for j, _ in runs[1:]:
                    assert labels[j] == label
                lrs = [list(zip(*run[1]['lr'])) for run in runs]
                X = lrs[0][0]
                for x, _ in lrs[1:]:
                    if len(x) > len(X):
                        X = x
                Y = []
                for j in range(len(X)):
                    cur = []
                    for _, lr in lrs:
                        if j < len(lr):
                            cur.append(lr[j])
                    Y.append(np.mean(cur))
                ax_lr.plot(
                    X, Y,
                    linestyle=':', linewidth=0.5, color=cmap(i)
                )

    ##########
    all_values = []
    times = {}
    testl = {}
    ngs = {}
    for dataset in set_config:
        max_epoch_size = 1
        if not group_batches:
            max_epoch_size = max(
                len(ys)
                for result in results
                for _, ys in result[dataset]
            )
        if plot_args.average_runs:
            _grouped_runs = {}
            for i, run in enumerate(results):
                # _l = len(run['train_losses'])
                # if _l == 78 or _l == 79:
                #     continue
                seed = run['config'].seed
                # if seed != 42:
                #     continue
                run['config'].seed = None
                hs = hashlib.sha256(json.dumps(
                    vars(run['config']),
                    sort_keys=True
                ).encode('utf-8')).hexdigest()
                run['config'].seed = seed
                if hs not in _grouped_runs:
                    _grouped_runs[hs] = []
                _grouped_runs[hs].append((i, run))
            grouped_runs = []
            for hs, runs in _grouped_runs.items():
                label = labels[runs[0][0]]
                for i, _ in runs[1:]:
                    assert labels[i] == label
                tmp_args = copy.deepcopy(set_config[dataset]['line'])
                if dataset == 'train_losses':
                    tmp_args['label'] = label
                grouped_runs.append((
                    tmp_args,
                    [run[dataset] for _, run in runs],
                    label,
                    np.array([
                        [
                            t * run.get('n_gpus', -1)
                            for t in run['epoch_time']
                        ]
                        for _, run in runs
                    ]),
                    [run.get('n_gpus', -1) for _, run in runs]
                ))
            runs = grouped_runs
        else:
            runs = []
            for label, result in zip(labels, results):
                # if result['config'].seed != 4242:
                #     continue
                tmp_args = copy.deepcopy(set_config[dataset]['line'])
                if dataset == 'train_losses':
                    tmp_args['label'] = label
                runs.append((
                    tmp_args, result[dataset], label,
                    np.array(result['epoch_time']) * result['n_gpus'],
                    [result['n_gpus']]
                ))
        for i_run, (tmp_args, result, label, ets, ng) in enumerate(runs):
            plot_fn = plot_average if plot_args.average_runs else plot_single
            id = -1
            if dataset == 'test_losses':
                id //= exp_args.runs[0].test_frequency
            tmp_r = result
            if not plot_args.average_runs:
                tmp_r = [tmp_r]
            losses = np.array([
                r[id if id < len(r) else -1][1][i_loss].mean()
                for r in tmp_r
            ])
            et = ets.mean()
            # if plot_args.average_runs and label[:7] == 'Cluster':
            #     print(f'Warning, [{label}] cheats!')
            #     if loss == 'accuracy':
            #         mask = losses > 0.7
            #     elif loss == 'BCE':
            #         mask = losses < 0.5
            #     else:
            #         assert False
            #     losses = losses[mask]
            #     result = [r for r, m in zip(result, mask) if m]
            id_mean = np.mean(losses)
            id_std = np.std(losses)
            print(f'{loss} {i_run} {label} {dataset} {id_mean} {id_std} {et:.2f}s')
            times[label] = et, np.std(ets), ets
            ngs[label] = ng
            if loss not in testl:
                testl[loss] = {}
            testl[loss][label] = (id_mean, id_std)
            values = plot_fn(
                ax, tmp_args, i_run, i_loss, result, max_epoch_size,
                group_batches, smooth_window=plot_args.smooth_window,
                ci_alpha=set_ci_alpha[dataset], x_max=plot_args.x_max
            )
            all_values.append(values)
    times2 = {}
    for lb in times:
        if lb[:2] == 'GN':
            continue
        times2[lb] = times[lb]
    for lb in times:
        if lb[:2] != 'GN':
            continue
        times2[lb] = times[lb]
    times = times2
    print(' '*20, end=' ')
    for lb in times:
        lbl = lb.split()[0]
        lbl += f'[{','.join(map(str, ngs[lb]))}]'
        print(f'{lbl:<20}', end=' ')
    print()
    print(f'{'time':<20}', end=' ')
    for lb in times:
        t = f'{times[lb][0]:>4.1f}\\pm{times[lb][1]:>4.1f}s'
        print(f'{t:<20}', end=' ')
    print()
    for loss in testl:
        print(f'{loss:<20}', end=' ')
        for lb in times:
            l = f'{testl[loss][lb][0]:.3f}\\pm{testl[loss][lb][1]:.3f}'
            print(f'{l:<20}', end=' ')
        print()
    print('#' * 80)
    print(f'{'model':<40}', end=' ')
    for loss in testl:
        print(f'{loss:<20}', end=' ')
    print('time')
    for lb in times:
        lbl = lb.split()[0]
        lbl += f'[{','.join(map(str, ngs[lb]))}]'
        print(f'{lbl:<40}', end=' ')
        for loss in testl:
            l = f'{testl[loss][lb][0]:.3f}\\pm{testl[loss][lb][1]:.3f}'
            print(f'{l:<20}', end=' ')
        t = f'{times[lb][0]:>4.1f}\\pm{times[lb][1]:>4.1f}s'
        print(f'{t}')
        if times[lb][1] > 2:
            print(times[lb][2].mean(axis=-1))
    all_values = np.concatenate(all_values)
    if plot_args.exclude_outliers:
        p = 0.05
        y_min, y_max = np.quantile(all_values, [p, 1-p])
        y_mean = (y_min + y_max) / 2
        y_delta = y_mean - y_min
        margin = 1.05
        y_min = y_mean - margin * y_delta
        y_max = y_mean + margin * y_delta
    else:
        y_min = all_values.min()
        y_max = all_values.max()
    ####################

    ax.set_xlim(
        1 if group_batches else 0,
        plot_args.x_max or max_epochs
    )
    if plot_args.y_max is not None and not np.isnan(plot_args.y_max[i_loss]):
        y_max = plot_args.y_max[i_loss]
    if plot_args.y_min is not None and not np.isnan(plot_args.y_min[i_loss]):
        y_min = plot_args.y_min[i_loss]
    if not np.isnan(y_max):
        ax.set_ylim(top=y_max)
    if not np.isnan(y_min):
        ax.set_ylim(bottom=y_min)

    if plot_args.mark_lines:
        for y_pos in ax.get_yticks():
            ax.axhline(y_pos, color='0.85', linestyle='--', linewidth=0.5)

    ax.legend()
    fig.savefig(
        path.parent / f'{'epoch' if group_batches else 'batch'}_{loss}.png',
        dpi=plot_args.dpi
    )


def main(plot_args=None):
    if plot_args is None:
        plot_args = parse_plot()

    from utils.gpu_management import set_gpus
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    set_gpus(1, forcing=False)

    path = pathlib.Path(plot_args.data)
    with open(path, 'rb') as f:
        exp_args, results = pickle.load(f)

    global cmap
    cmap = plt.get_cmap(plot_args.cmap)
    plt.rcParams['font.family'] = 'monospace'

    labels = make_labels(
        [run['config'] for run in results],
        plot_args.show_param_count,
        plot_args.average_runs
    )

    plot(exp_args, labels, results, plot_args, path)
    if not plot_args.skip_batch:
        plot(exp_args, labels, results, plot_args, path, group_batches=False)

if __name__ == '__main__':
    main()
