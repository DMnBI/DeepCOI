# Adapted from https://fairseq.readthedocs.io/en/latest/_modules/fairseq/optim/lr_scheduler/inverse_square_root_schedule.html

from torch.optim.lr_scheduler import LambdaLR


class InverseSqrtScheduler(LambdaLR):
    """ Linear warmup and then follows an inverse square root decay schedule
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Afterward, learning rate follows an inverse square root decay schedule.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)
