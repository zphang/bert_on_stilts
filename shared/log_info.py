# Todo: optionally use logger


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def print_trainable_params(model):
    print("TRAINABLE PARAMS:")
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print("    {}  {}".format(param_name, tuple(param.shape)))
