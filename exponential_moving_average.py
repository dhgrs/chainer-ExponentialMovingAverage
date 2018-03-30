import chainer
from chainer import configuration
from chainer import link


class ExponentialMovingAverage(link.Chain):

    def __init__(self, target, decay=0.999):
        super(ExponentialMovingAverage, self).__init__()
        self.decay = decay
        with self.init_scope():
            self.target = target
            for name, param in target.namedparams():
                setattr(self, 'ema' + name, chainer.Parameter(param.array))

    def __call__(self, *args, **kwargs):
        if configuration.config.train:
            ys = self.target(*args, **kwargs)
            with self.init_scope():
                for name, param in self.target.namedparams():
                    ema_name = 'ema' + name
                    if not param.requires_grad or param.array is None:
                        new_average = param.array
                    else:
                        new_average = self.decay * param.array + \
                            (1 - self.decay) * self[ema_name].array
                    setattr(self, ema_name, chainer.Parameter(new_average))
        else:
            # TODO: use exponential moving average when train is False
            ys = self.target(*args, **kwargs)
        return ys
