from ggnn.data import make_dataset
from ggnn.data.ogbg import DSMolhiv

def test_ogbg_molhiv():
    ds_train = make_dataset(DSMolhiv, cache=False)
    ds_valid = make_dataset(DSMolhiv, category='valid', cache=False)
    ds_test = make_dataset(DSMolhiv, category='test', cache=False)
    assert ds_train.n_samples == 32901
    assert ds_valid.n_samples == 4113
    assert ds_test.n_samples == 4113
