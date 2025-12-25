import os
import pickle
from typing import TypeVar
import hashlib
import json

from ggnn.data.dataset import Dataset

CSI = "\x1b[" #]
D = TypeVar("D", bound=Dataset)

def make_dataset(
    dataset_cls: type[D],
    cache=True,
    **kwargs
) -> D:
    args = locals()
    del args['cache']
    args['dataset_cls'] = dataset_cls.__name__
    hs = hashlib.sha256(
        json.dumps(args, sort_keys=True).encode('utf-8')
    ).hexdigest()
    cache_file = f'cache/dataset/{args['dataset_cls']}/{hs}.pkl'
    if cache:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cached_args, dataset = pickle.load(f)
                if cached_args != args:
                    raise ValueError(
                        f'Cached dataset has different arguments\n'
                        f'{CSI}31m{args}{CSI}0m\n'
                        f'{CSI}34m{cached_args}{CSI}0m'
                    )
                return dataset_cls.from_serialized(dataset)

    dataset = dataset_cls(**kwargs)

    if cache:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((args, dataset.to_serializable()), f)

    return dataset
