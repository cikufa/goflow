"""Load the released optional central-value model, including normalization state."""
from rl_games.algos_torch import model_builder


def privileged_value_model(checkpoint, central_config, device='cuda:0'):
    weights = checkpoint['assymetric_vf_nets']
    size = weights['model.a2c_network.actor_mlp.0.weight'].shape[1]
    builder = model_builder.ModelBuilder()
    definition = builder.load({'model': {'name': 'central_value'},
                               'network': central_config['network']})
    model = definition.build({'input_shape': (size,), 'actions_num': 3, 'num_seqs': 1,
                              'value_size': 1, 'normalize_input': central_config['normalize_input'],
                              'normalize_value': central_config['normalize_value']}).to(device)
    model.load_state_dict({key.removeprefix('model.'): value for key, value in weights.items()})
    model.eval()
    return model
