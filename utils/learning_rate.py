import numpy as np

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, 40.0)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(args, optimizer, i_iter):
    if i_iter >= args.max_iter:
        i_iter = args.max_iter

    lr = lr_poly(args.base_lr, i_iter, args.max_iter, args.power)  # 这里的args.max_iter会引发一个bug
    lr = max(5e-6,lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_learning_rate_section(args, optimizer, i_iter):

    if i_iter <= 10000:
        lr = lr_poly(args.base_lr, i_iter, args.max_iter, args.power)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    if i_iter > 10000:

        if i_iter >= 29999:
            i_iter = 29999

        lr = lr_poly(0.00001, i_iter - 10000, 30000, args.power)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
    return lr

def adjust_learning_rate_separate(args, optimizer_seg, optimizer_cls, i_iter):
    if i_iter >= args.max_iter:
        i_iter = args.max_iter

    lr = lr_poly(args.base_lr, i_iter, args.max_iter, args.power)  # 这里的args.max_iter会引发一个bug

    for param_group in optimizer_seg.param_groups:
        # print(len(param_group["params"]))
        param_group["lr"] = lr

    for param_group in optimizer_cls.param_groups:
        # print(len(param_group["params"]))
        param_group["lr"] = lr / 10

    # optimizer.param_groups[0]['lr'] = lr
    return lr


def warm_up_decay_learning_rate(args, optimizer, i_iter, warmup_iter):
    if i_iter >= args.max_iter:
        i_iter = args.max_iter

    if i_iter <= warmup_iter:
        lr = (0.0001 - 0.00001) / warmup_iter * i_iter + 0.00001
    else:
        lr = lr_poly(args.base_lr, i_iter - warmup_iter + 1, args.max_iter, args.power)  # 这里的args.max_iter会引发一个bug
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # optimizer.param_groups[0]['lr'] = lr
    return lr

def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr