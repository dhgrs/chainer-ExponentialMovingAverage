import numpy
import chainer
import chainer.links as L
from exponential_moving_average import ExponentialMovingAverage

# input and output example
x = numpy.arange(10 * 4, dtype=numpy.float32).reshape((10, 4))
y = numpy.arange(10 * 20, dtype=numpy.float32).reshape((10, 20))

# initial parameters
initialW = numpy.arange(4 * 20, dtype=numpy.float32).reshape((20, 4))
initial_bias = numpy.zeros(20, dtype=numpy.float32)

# define networks
with_ema = ExponentialMovingAverage(
    L.Linear(4, 20, initialW=initialW, initial_bias=initial_bias), decay=0.5)
opt_with_ema = chainer.optimizers.SGD()
opt_with_ema.setup(with_ema)

without_ema = L.Linear(4, 20, initialW=initialW, initial_bias=initial_bias)
opt_without_ema = chainer.optimizers.SGD()
opt_without_ema.setup(without_ema)

# print initial parameters
print('initial parameters')
for name, param in with_ema.namedparams():
    print('with ema:', name, param[0])
for name, param in without_ema.namedparams():
    print('without ema:', name, param[0])

# test exponential moving average
for i in range(5):
    print('')
    print('iteration', i)
    # calculate losses and update networks
    loss = chainer.functions.mean_squared_error(with_ema(x), y)
    with_ema.cleargrads()
    loss.backward()
    opt_with_ema.update()

    loss = chainer.functions.mean_squared_error(without_ema(x), y)
    without_ema.cleargrads()
    loss.backward()
    opt_without_ema.update()

    # print updated parameters
    for name, param in with_ema.namedparams():
        print('with ema:', name, param[0])
    for name, param in without_ema.namedparams():
        print('without ema:', name, param[0])
