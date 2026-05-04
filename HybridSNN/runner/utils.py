def reset_states(model):
    """Call `reset()` on every submodule that provides it."""
    for m in model.modules():
        if hasattr(m, 'reset'):
            m.reset()
