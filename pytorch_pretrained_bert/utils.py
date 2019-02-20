def to_cpu(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, dict):
            new_state_dict[k] = to_cpu(v)
        elif isinstance(v, int):
            new_state_dict[k] = v
        elif isinstance(v, list):
            # May need to change this in the future
            assert k == "param_groups"
            new_state_dict[k] = v
        else:
            new_state_dict[k] = v.cpu()
    return new_state_dict


def only_one_of(ls):
    return count_bool(ls) == 1


def at_most_one_of(ls):
    return count_bool(ls) <= 1


def count_bool(ls):
    return sum([1 if elem else 0 for elem in ls])
