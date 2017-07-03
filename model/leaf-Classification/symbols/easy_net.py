import mxnet as mx

def get_symbol(num_classes, **kwargs):
    data = mx.symbol.Variable(name='data')
    conv1 = mx.symbol.Convolution(name='conv1', data=data, num_filter=32, kernel=(5,5))
    relu1 = mx.symbol.Activation(name='relu1', data=conv1, act_type='relu')
    pool1 = mx.symbol.Pooling(name='pool1', data=relu1, kernel=(2,2), stride=(2,2), pool_type='max')

    conv2 = mx.symbol.Convolution(name='conv2', data=data, num_filter=64, kernel=(5,5))
    relu2 = mx.symbol.Activation(name='relu2', data=conv2, act_type='relu')
    pool2 = mx.symbol.Pooling(name='pool2', data=relu2, kernel=(2,2), stride=(2,2), pool_type='max')

    features = mx.symbol.Variable(name='features')
    flatten = mx.symbol.Flatten(name='flatten', data=pool2)
    concat = mx.symbol.Concat(name='concat', *[flatten, features])
    fc1 = mx.symbol.FullyConnected(data=concat, num_hidden=100, name='fc1')
    fc2 = mx.symbol.FullyConnected(data=fc1, num_hidden=num_classes, name='fc2')

    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return softmax

