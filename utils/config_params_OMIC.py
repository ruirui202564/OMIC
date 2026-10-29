def config_OMIC():
    params_list = {'alpha': [15],
                   'max_w':[50.0],
                   'win_label':[1500]}
    return params_list

def create_param_OMIC(model_params):
    param_combination = []
    for a in model_params['alpha']:
            for c in model_params['max_w']:
                for d in model_params['win_label']:
                    param_combination.append({'alpha': a, 'max_w': c, 'win_label': d})
    return param_combination