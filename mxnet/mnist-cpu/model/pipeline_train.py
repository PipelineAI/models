import mxnet as mx
mnist = mx.test_utils.get_mnist()

# Fix the seed
mx.random.seed(42)

# Set the compute context, GPU is available otherwise CPU
ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()

batch_size = 1024
train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

lenet_model = mx.mod.Module(symbol=lenet, context=ctx)
num_epoch = 1
# train with the same
lenet_model.fit(train_iter,
                eval_data=val_iter,
                optimizer='sgd',
                optimizer_params={'learning_rate':0.1},
                eval_metric='acc',
                batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                num_epoch=num_epoch)
print(lenet_model)
prefix = 'mnist'

lenet_model.save_checkpoint(prefix, num_epoch)

# PREDICTION
# load model back
symbol, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_epoch)
lenet_model_loaded = mx.mod.Module(symbol=symbol)
lenet_model_loaded.set_params(arg_params=arg_params, aux_params=aux_params)

test_iter = mx.io.NDArrayIter(mnist['test_data'], None, batch_size)
#prob = lenet_model.predict(test_iter)
prob = lenet_model_loaded.predict(test_iter)
test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

# predict accuracy for lenet
acc = mx.metric.Accuracy()
lenet_model_loaded.score(test_iter, acc)
print(acc)
assert acc.get()[1] > 0.98, "Achieved accuracy (%f) is lower than expected (0.98)" % acc.get()[1]
