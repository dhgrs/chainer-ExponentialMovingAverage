import copy

import chainer
from chainer import configuration
from chainer import link


class ExponentialMovingAverage(link.Chain):

    def __init__(self, target, decay=0.999):
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        with self.init_scope():
            self.target = target
            self.ema = copy.deepcopy(target)

    def __call__(self, *args, **kwargs):
        if configuration.config.train:
            ys = self.target(*args, **kwargs)
            for target_name, target_param in self.target.namedparams():
                for ema_name, ema_param in self.ema.namedparams():
                    if target_name == ema_name:
                        if not target_param.requires_grad \
                                or ema_param.array is None:
                            new_average = target_param.array
                        else:
                            new_average = self.decay * target_param.array + \
                                (1 - self.decay) * ema_param.array
                        ema_param.array = new_average
        else:
            ys = self.ema(*args, **kwargs)
        return ys
