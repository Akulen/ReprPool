from functools import partial

from ggnn.models.gnn import *

def get_node_model(model_name: str):
    match model_name:
        case 'GraphUNet':
            return GraphUNetNodeClassifier
        case 'ReG':
            return ReGNodeClassifier
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_model(model_name: str, task: str='graph'):
    if task == 'node':
        return get_node_model(model_name)
    assert task == 'graph'
    match model_name:
        case 'GINNet':
            return GINNet
        case 'GIN':
            raise NotImplementedError
            # return GINClassifier
        case 'GatedGCNNet':
            return GatedGCNNet
        case 'UFNet':
            return UFNet
        case 'GNNet':
            return GNNet
        case 'GNNet2':
            return GNNet2
        # case 'UF':
        #     return UFClassifier
        case 'ClusterPooling':
            return ClusterPoolingClassifier
        case 'UFPooling':
            return UFPoolingClassifier
        case 'GNPooling':
            return GNPoolingClassifier
        # TODO: GAT, GATConv, GATEAU
        case 'GNClustering':
            return GNClustering
        case 'TopKPool':
            return TopKPoolClassifier
        case 'SAGPool':
            return SAGPoolClassifier
        case 'MinCutPool':
            return MinCutPoolClassifier
        case 'DiffPool':
            return DiffPoolClassifier
        case _:
            raise ValueError(f"Unknown model: {model_name}")

def get_model_params(model_name: str, task: str='graph'):
    import inspect
    return inspect.signature(get_model(model_name, task)).parameters
