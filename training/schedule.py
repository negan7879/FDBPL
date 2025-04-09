import math


def lr_setter(optimizer, epoch, args, bl=False):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # lr = args.lr
    lr = args.lrbl * (0.1 ** (epoch // (args.epochb * 0.5)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
