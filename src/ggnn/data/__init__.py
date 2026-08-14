import os
import pickle
import hashlib
import json
from functools import partial

from ggnn.data.dataset import Dataset
from ggnn.data.ogbg import DSMolhiv
from ggnn.data.synth import DSSynth
from ggnn.data.mnist import DSMnist
from ggnn.data.cifar10 import DSCifar10
from ggnn.data.tudataset import DSRedditB, DSRedditM5K, DSRedditM12K, DSNCI1, DSCOLLAB
from ggnn.data.node import DSCora, DSCiteseer, DSPubMed

from benchmarking.data.superpixels import SuperPixDataset

def get_dataset(name: str) -> type[Dataset]:
    match name:
        case 'ogbg-molhiv':
            return DSMolhiv
        case 'synth':
            return DSSynth
        case 'mnist':
            return DSMnist
        case 'cifar10':
            return DSCifar10
        case 'rdt-b':
            return DSRedditB
        case 'rdt-m5k':
            return DSRedditM5K
        case 'rdt-m12k':
            return DSRedditM12K
        case 'nci1':
            return DSNCI1
        case 'collab':
            return DSCOLLAB
        case 'cora':
            return DSCora
        case 'citeseer':
            return DSCiteseer
        case 'pubmed':
            return DSPubMed
        case _:
            raise ValueError(f'Unknown dataset: {name}')

CSI = "\x1b[" #]

def make_dataset(
    dataset_cls: str | type[Dataset],
    cache=True,
    **kwargs
) -> Dataset:
    if isinstance(dataset_cls, str):
        dataset_cls = get_dataset(dataset_cls)
    assert(isinstance(dataset_cls, type) and issubclass(dataset_cls, Dataset))
    args = locals()
    del args['cache']
    if 'verbose' in args['kwargs']:
        del args['kwargs']['verbose']
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
