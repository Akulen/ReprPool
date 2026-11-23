import argparse
import copy
import json
import sys
import time
import yaml

def override_args(
    args: dict, override: dict | None, path: list | None=None
) -> None:
    if override is None:
        return
    if path is None:
        path = []
    for k, v in override.items():
        if len(path) == 0 and k not in args:
            raise ValueError(f"Unknown argument: {k}")
        if isinstance(v, dict):
            if k not in args:
                args[k] = {}
            if isinstance(args[k], str):
                args[k] = json.loads(args[k])
            override_args(args[k], v, path+[k])
        else:
            args[k] = v

def parse_exp():
    parser = argparse.ArgumentParser(
        description="Global GNN Experiments "
        "(Starred arguments can be lists in config file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Experiment Metadata
    group_meta = parser.add_argument_group('Experiment Metadata')
    group_meta.add_argument('--name',
        type=str, default='<current datetime>', help='Experiment Name'
    )
    group_meta.add_argument('--n-gpus','-d',
        type=int, default=2, help='Number of GPUs'
    )
    group_meta.add_argument('--test-frequency', type=int, default=5)
    group_meta.add_argument('--seed', type=int, default=42, help='*Random seed')

    # Experiment Config
    group_data = parser.add_argument_group('Data Config')
    group_data.add_argument('--dataset',
        type=str, default='ogbg-molhiv', choices=['ogbg-molhiv', 'cora'],
        help='Dataset'
    )

    group_archi = parser.add_argument_group('Model Architecture')
    group_archi.add_argument('--model', '-m',
        type=str, default='GIN', choices=[
            'GIN', 'GraphUNet'
        ],
        help='*Model'
    )
    group_archi.add_argument('--model-params', '-mp', type=str, default='{}',
        help='Model parameters as a JSON dictionary string, leave empty to use '
             'defaults, use \'h\' instead to list available parameters'
    )
    group_archi.add_argument('--models', type=str, default='[]',
        help='List of model configs. Overrides other settings when defined, '
             'except lists.'
    )

    group_train = parser.add_argument_group('Training Config')
    group_train.add_argument('--epochs',
        type=int, default=10, help='*Number of epochs'
    )
    group_train.add_argument('--batch-size', '-bs',
        type=int, default=32, help='*Batch size')
    group_train.add_argument('--vt-batch-size',
        type=int, default=-1,
        help='Validation and Test Batch size. If 0 use regular batch size. '
             'If -1 use full dataset.'
    )
    group_train.add_argument('--optimizer',
        type=str, default='adamw', choices=['adam', 'adamw', 'lion'],
        help='Optimizer'
    )
    group_train.add_argument('--lr',
        type=float, default=1e-6, help='*Learning rate'
    )
    group_train.add_argument('--weight-decay',
        type=float, default=0.0, help='Weight decay'
    )
    group_train.add_argument('--scheduler',
        type=str, default='constant', choices=['constant', 'warmup_cosine'],
        help='Scheduler'
    )
    group_train.add_argument('--scheduler-params', type=str, default='{}')
    group_train.add_argument('--loss',
        action='extend', nargs='+', type=str,
        choices=['BCE', 'NLL', 'CrossEntropy', 'accuracy'], default=[],
        help='Tracked loss. First one is the optimized loss. (Default: BCE)'
    )

    # Other
    group_other = parser.add_argument_group('Other')
    group_other.add_argument('--force',
        action='store_true', help='Ignore cached runs'
    )
    group_other.add_argument('--debug', action='store_true')
    group_other.add_argument('--conf',
        action='append', help='Use configuration file'
    )
    group_other.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args()
    CSI = "\x1b[" #]

    if args.conf is not None:
        for conf_fname in args.conf:
            with open(conf_fname, 'r') as f:
                conf = yaml.safe_load(f)
                for k in conf:
                    if parser.get_default(k) is None:
                        raise ValueError(f'Unknown parameter {k}')
                parser.set_defaults(**conf)
        # Reload arguments to override config file values with command line
        # values
        args = parser.parse_args()

    if args.name == '<current datetime>':
        args.name = time.strftime("%Y%m%d-%H%M%S")
    if args.loss is None or len(args.loss) == 0:
        args.loss = ['BCE']
    if 'dataset' in args.loss:
        if len(args.loss) == 1:
            raise ValueError('Cannot optimize on dataset loss')
        args.loss.remove('dataset')
        args.loss.append('dataset')
    if isinstance(args.models, str):
        args.models = json.loads(args.models)

    exp_args_hp = [
        'name', 'n_gpus', 'dataset', 'loss', 'verbose', 'conf', 'debug', 'force'
    ]
    run_args_hp = ['dataset', 'loss']

    exp_args = copy.deepcopy(args)
    for hp in vars(args):
        if not hp in exp_args_hp:
            del vars(exp_args)[hp]

    for hp in exp_args_hp:
        if hp not in run_args_hp:
            del vars(args)[hp]

    models = args.models
    del args.models #type: ignore
    if len(models) > 0:
        runs = []
        for overrides in models:
            run = copy.deepcopy(args)
            if isinstance(run, str):
                run = json.loads(run)
            override_args(vars(run), overrides)
            runs.append(run)
    else: runs = [copy.deepcopy(args)]

    arg_lists = ['seed', 'model', 'epochs', 'batch_size', 'lr']
    for hp in arg_lists:
        if isinstance(vars(args)[hp], list):
            base_runs = runs
            runs = []
            for run in base_runs:
                for val in vars(run)[hp]:
                    new_run = copy.deepcopy(run)
                    vars(new_run)[hp] = val
                    runs.append(new_run)

    hyperparameters = {
        'float': ['lr', 'weight_decay'],
        'json': ['scheduler_params', 'model_params']
    }
    for run in runs:
        for hp in hyperparameters['float']:
            if isinstance(vars(run)[hp], str):
                vars(run)[hp] = float(vars(run)[hp])
        if run.model_params == 'h':
            from ggnn.models import get_model_params
            print(
                f'{CSI}34mAvailable parameters for '
                f'{CSI}32m{run.model}{CSI}34m:{CSI}0m'
            )
            for param, sign in get_model_params(run.model).items():
                if param in ['self', 'key', 'cards']: continue
                print(
                    f'{CSI}32m{param}{CSI}34m: '
                    f'{CSI}33m{str(sign)[len(str(param))+2:]}{CSI}0m'
                )
            sys.exit()
        for hp in hyperparameters['json']:
            if isinstance(vars(run)[hp], str):
                vars(run)[hp] = json.loads(vars(run)[hp])
    if exp_args.verbose > 0:
        print(exp_args)
    exp_args.runs = runs

    return exp_args

def parse_auto_exp():
    parser = argparse.ArgumentParser(
        description="AutoML Hyperparameter Search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--name', type=str, default='<current datetime>')
    parser.add_argument('--n-gpus', '-d', type=int, default=2)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--model', '-m', type=str, default='GraphUNet')
    parser.add_argument('--model-params', '-mp', type=str, default='{}')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='constant')
    parser.add_argument('--scheduler-params', type=str, default='{}')
    parser.add_argument('--loss',
        action='extend', nargs='+', type=str, default=[]
    )
    parser.add_argument('--lr', type=str, default='1e-3')
    parser.add_argument('--weight-decay', type=str, default='0')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--seed', type=str, default='[42]')
    parser.add_argument('--optuna-storage', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--conf', action='append')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--batch-size', '-bs', type=int, default=32)
    parser.add_argument('--vt-batch-size', type=int, default=-1)
    parser.add_argument('--test-frequency', type=int, default=5)

    args = parser.parse_args()

    if args.conf is not None:
        for conf_fname in args.conf:
            with open(conf_fname, 'r') as f:
                conf = yaml.safe_load(f)
            parser.set_defaults(**conf)
        args = parser.parse_args()

    if args.name == '<current datetime>':
        args.name = time.strftime("%Y%m%d-%H%M%S")
    if not args.loss:
        args.loss = ['BCE']
    if isinstance(args.seed, str):
        args.seed = json.loads(args.seed)
    if isinstance(args.lr, str):
        args.lr = json.loads(args.lr)
    if isinstance(args.lr, list):
        args.lr = list(map(float, args.lr))
    if isinstance(args.weight_decay, str):
        args.weight_decay = json.loads(args.weight_decay)
    if isinstance(args.weight_decay, list):
        args.weight_decay = list(map(float, args.weight_decay))
    if isinstance(args.model_params, str):
        args.model_params = json.loads(args.model_params)
    if isinstance(args.scheduler_params, str):
        args.scheduler_params = json.loads(args.scheduler_params)

    return args

def parse_plot():
    parser = argparse.ArgumentParser(
        description="Plotter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('data', type=str)
    group_meta = parser.add_argument_group('Plot Config')
    group_meta.add_argument('--width', type=int, default=16)
    group_meta.add_argument('--height', type=int, default=9)
    group_meta.add_argument('--dpi', type=int, default=300)
    group_meta.add_argument('--cmap', type=str, default='tab10')
    group_meta.add_argument('--y-max', type=float, nargs='+', default=None)
    group_meta.add_argument('--y-min', type=float, nargs='+', default=None)
    group_meta.add_argument('--x-max', type=int, default=None)
    group_meta.add_argument('--exclude-outliers',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--only-test-indicators',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--mark-lines',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--plot-lr',
        default=True, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--average-runs',
        default=False, action=argparse.BooleanOptionalAction,
        help='Average over runs with the same config but different seeds'
    )
    group_meta.add_argument('--smooth-window', type=int, default=1)
    group_meta.add_argument('--focus-valid',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--focus-test',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--skip-batch',
        default=False, action=argparse.BooleanOptionalAction
    )
    group_meta.add_argument('--show-param-count',
        default=False, action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()
    print(args)
    assert args.smooth_window > 0 and args.smooth_window % 2 == 1
    return args
