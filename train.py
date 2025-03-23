from argparse import ArgumentParser
from accelerate import Accelerator, DistributedDataParallelKwargs


def train_exchange(conf, accl):
    from model.trainer_uncond import TrainerUnCond
    train = TrainerUnCond(conf, accl=accl)
    return train.main()


def main(args):
    ununsed = args.force_unused
    kwgs = DistributedDataParallelKwargs(find_unused_parameters=ununsed)

    if args.force_cpu:
        accl = Accelerator(cpu=True, kwargs_handlers=[kwgs])
    else:
        accl = Accelerator(kwargs_handlers=[kwgs])

    if args.task == 'uncond':
        train_exchange(args.config, accl)
    elif args.task == 'cond':
        raise NotImplementedError
    else:
        print("task {} not recognized1".format(args.task))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--task", choices=['uncond'], default='uncond')
    parser.add_argument("--config", type=str, default="./configs/conf_exchange.toml")
    parser.add_argument("--force_cpu", action='store_true')
    parser.add_argument("--force_unused", action='store_true')

    args = parser.parse_args()
    main(args)
