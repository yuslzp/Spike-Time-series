def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()
