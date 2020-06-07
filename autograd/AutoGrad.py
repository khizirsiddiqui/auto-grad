from dualnumber import DualNumber

def derivative(fn, x, args=None):
    if args is None:
        return fn(DualNumber(x, 1)).b

    fn_args = []
    for i, arg in enumerate(args):
        if i == x:
            fn_args.append(DualNumber(arg, 1))
        else:
            fn_args.append(DualNumber(arg, 0))

    return fn(*fn_args).b
