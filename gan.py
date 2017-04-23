import mxnet as mx
import numpy as np
from sklearn.datasets import fetch_mldata
import logging
import cv2
from datetime import datetime

def make_sym(ngf, ndf, nc, is_wgan = True, image_size=28, is_mlp=False, no_bias=True, fix_gamma=True, eps=1e-5 + 1e-12):
    BatchNorm = mx.sym.BatchNorm
    cond = mx.sym.Variable('cond')
    rand = mx.sym.Variable('rand')

    cond = mx.sym.Reshape(cond, shape=(-1, 10, 1, 1))

    mixed = mx.sym.Concat(cond, rand, dim=1)

    #deconvolution: osize = stride * (isize - 1) + ksize - 2 * pad + adj

    g1 = mx.sym.Deconvolution(mixed, name='g1', kernel=(4,4), num_filter=ngf*4, no_bias=no_bias)
    gbn1 = BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=eps)
    gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')
    #4x4

    g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ngf*2, no_bias=no_bias)
    gbn2 = BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=eps)
    gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')
    #8x8

    g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(2,2), num_filter=ngf, no_bias=no_bias)
    gbn3 = BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=eps)
    gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')
    #14x14

    g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=nc, no_bias=no_bias)
    gout = mx.sym.Activation(g4, name='gact4', act_type='tanh')
    #28x28

    data = mx.sym.Variable('data')
    condD = mx.sym.Variable('condD')

    flatten_data = mx.sym.Flatten(data=data)
    input_data = mx.sym.Concat(flatten_data, condD)

    if is_mlp:
        net = mx.sym.FullyConnected(input_data, num_hidden=512, name='fc1')
        net = mx.sym.Activation(net, name='relu1', act_type="relu")
        net = mx.sym.FullyConnected(net, num_hidden=512, name='fc2')
        net = mx.sym.Activation(net, name='relu2', act_type="relu")
        net = mx.sym.FullyConnected(net, num_hidden=500, name="fc3")
        net = mx.sym.Activation(net, act_type="relu")
        net = mx.sym.FullyConnected(net, num_hidden=500, name="fc4")
        net = mx.sym.Activation(net, act_type="relu")
        net = mx.sym.FullyConnected(net, num_hidden=500, name="fc5")
        net = mx.sym.Activation(net, act_type="relu")
        d5 = mx.sym.FullyConnected(net, num_hidden=1, name="fc6")
        if not is_wgan:
            d5 = mx.sym.Activation(d5, act_type="sigmoid")
    else:
        d0 = mx.sym.FullyConnected(input_data, name="g1", num_hidden=image_size*image_size, no_bias=no_bias)
        d0 = mx.sym.Activation(d0, name="gact1", act_type="relu")
        d0 = mx.sym.Reshape(d0, name="gout", shape=(-1, 1, image_size,image_size))

        d1 = mx.sym.Convolution(d0, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf, no_bias=no_bias)
        dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

        d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*2, no_bias=no_bias)
        dbn2 = BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=eps)
        dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

        d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=ndf*4, no_bias=no_bias)
        dbn3 = BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=eps)
        dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

        d5 = mx.sym.Flatten(dact3)
        d5 = mx.sym.FullyConnected(d5, num_hidden=1, name="fc_dloss")
        if not is_wgan:
            d5 = mx.sym.Activation(d5, act_type="sigmoid")

    return gout, d5


class WGANMetric(object):
    ''' metric for wgan
    '''

    def __init__(self):
        self.update_ = 0
        self.value_ = 0

    def reset(self):
        '''reset status
        '''
        self.update_ = 0
        self.value_ = 0

    def update(self, val):
        '''update metric
        '''
        self.update_ += 1
        self.value_ += val

    def get(self):
        '''get metric value
        '''
        return self.value_ / self.update_



def get_mnist(image_size):
    mnist = fetch_mldata('MNIST original')
    np.random.seed(1234) # set seed for deterministic ordering
    p = np.random.permutation(mnist.data.shape[0])
    X = mnist.data[p]
    X = X.reshape((70000, 1, image_size, image_size))
    Y = mnist.target[p]

    X = X.astype(np.float32)/(255.0/2) - 1.0
    X_train = X[:60000]
    X_test = X[60000:]
    Y_train = Y[:60000]
    Y_test = Y[60000:]

    return X_train, X_test, Y_train, Y_test

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]

def fill_buf(buf, i, img, shape):
    n = buf.shape[0]/shape[1]
    m = buf.shape[1]/shape[0]

    sx = (i%m)*shape[0]
    sy = (i/m)*shape[1]
    buf[sy:sy+shape[1], sx:sx+shape[0], :] = img

def convert2img(X):
    assert len(X.shape) == 4
    X = X.transpose((0, 2, 3, 1))
    X = np.clip((X+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    n = np.ceil(np.sqrt(X.shape[0]))
    buff = np.zeros((int(n*X.shape[1]), int(n*X.shape[2]), X.shape[3]), dtype=np.uint8)
    for i, img in enumerate(X):
        fill_buf(buff, i, img, X.shape[1:3])
    return buff

def visual(title, X):
    buff = convert2img(X)
    cv2.imshow(title, buff)
    cv2.waitKey(1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    ndf = 64
    ngf = 64
    nc = 1
    batch_size = 100
    n_labels = 10
    Z = 100
    lr = 0.00005
    beta1 = 0.5
    ctx = mx.gpu(0)
    check_point = False
    wclip = 0.01
    image_size = 28
    is_mlp = False
    epoch_num = 50
    is_wgan = True

    symG, symD = make_sym(ngf, ndf, nc, is_wgan, image_size, is_mlp)

    # ==============data==============
    X_train, X_test, Y_train, Y_test = get_mnist(image_size)
    train_iter = mx.io.NDArrayIter(X_train, label=Y_train, batch_size=batch_size)
    rand_iter = RandIter(batch_size, Z)

    # =============module G=============
    modG = mx.mod.Module(symbol=symG, data_names=('cond','rand',), label_names=None, context=ctx)
    modG.bind(data_shapes=[('cond', (batch_size, n_labels))]+rand_iter.provide_data)
    modG.init_params(initializer=mx.init.Normal(0.02))
    modG.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods = [modG]

    # =============module D=============
    modD = mx.mod.Module(symbol=symD, data_names=('data','condD',), label_names=None, context=ctx)
    modD.bind(data_shapes=train_iter.provide_data+[('condD',(batch_size, n_labels))],
              inputs_need_grad=True)
    modD.init_params(initializer=mx.init.Normal(0.02))
    modD.init_optimizer(
        optimizer='adam',
        optimizer_params={
            'learning_rate': lr,
            'wd': 0.,
            'beta1': beta1,
        })
    mods.append(modD)

    def norm_stat(d):
        return mx.nd.norm(d)/np.sqrt(d.size)

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    if is_wgan:
        metricD = WGANMetric()
        metricG = WGANMetric()
    else:
        metricD = mx.metric.CustomMetric(fentropy)
        metricG = mx.metric.CustomMetric(fentropy)
        metricACC = mx.metric.CustomMetric(facc)

    print 'Training...'
    stamp =  datetime.now().strftime('%Y_%m_%d-%H_%M')

    one = mx.nd.ones((batch_size, 1), ctx) / batch_size
    mone = -mx.nd.ones((batch_size,1), ctx) / batch_size
    label = mx.nd.zeros((batch_size,1), ctx=ctx)

    # =============train===============
    for epoch in range(epoch_num):
        if is_wgan:
            for params in modD._exec_group.param_arrays:
                for param in params:
                    mx.nd.clip(param, -wclip, wclip, out=param)

        train_iter.reset()
        for t, batch in enumerate(train_iter):

            batch_label_one_hot = np.zeros((batch_size, n_labels), dtype=np.float32)
            batch_label_np = batch.label[0].asnumpy()
            for i in xrange(batch_size):
                batch_label_one_hot[i, int(batch_label_np[i])] = 1
            batch_label_one_hot = mx.nd.array(batch_label_one_hot)

            rbatch = rand_iter.next()
            modG.forward(mx.io.DataBatch([batch_label_one_hot]+rbatch.data, []), is_train=True)
            outG = modG.get_outputs()

            # update discriminator on fake
            modD.forward(mx.io.DataBatch(outG+[batch_label_one_hot], []), is_train=True)
            if is_wgan:
                errD_fake = modD.get_outputs()[0].asnumpy()
                modD.backward([mone])
            else:
                label[:] = 0
                diff = modD.get_outputs()[0] - label
                modD.backward([diff])
                metricD.update([label], modD.get_outputs())
                metricACC.update([label], modD.get_outputs())


            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in modD._exec_group.grad_arrays]

            # update discriminator on real
            modD.forward(mx.io.DataBatch(batch.data+[batch_label_one_hot], []), is_train=True)
            if is_wgan:
                errD_real = modD.get_outputs()[0].asnumpy()
                modD.backward([one])
            else:
                label[:] = 1
                diff = modD.get_outputs()[0] - label
                modD.backward([diff])
                metricD.update([label], modD.get_outputs())
                metricACC.update([label], modD.get_outputs())

            for gradsr, gradsf in zip(modD._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            modD.update()

            # update generator
            modD.forward(mx.io.DataBatch(outG+[batch_label_one_hot], []), is_train=True)
            if is_wgan:
                modD.backward([one])
            else:
                label[:] = 1
                diff = modD.get_outputs()[0] - label
                modD.backward([diff])
                metricG.update([label], modD.get_outputs())

            diffD = modD.get_input_grads()
            modG.backward([diffD[0]])

            modG.update()

            if is_wgan:
                errD = (errD_real - errD_fake) / batch_size
                metricD.update(errD.mean())
                errG = modD.get_outputs()[0] / batch_size
                metricG.update(errG.asnumpy().mean())

            t += 1
            if t % 10 == 0:
                if is_wgan:
                    print("epoch:", epoch+1, "iter:", t, "G: ", metricG.get(), "D: ", metricD.get())
                    metricD.reset()
                    metricG.reset()
                else:
                    print 'epoch:', epoch, 'iter:', t, 'metric:', metricACC.get(), metricG.get(), metricD.get()
                    metricACC.reset()
                    metricG.reset()
                    metricD.reset()

                batch_label_one_hot = np.zeros((batch_size, n_labels), dtype=np.float32)
                for i in xrange(batch_size):
                    index = i % 10
                    batch_label_one_hot[i, index] = 1
                batch_label_one_hot = mx.nd.array(batch_label_one_hot)

                rbatch = rand_iter.next()
                modG.forward(mx.io.DataBatch([batch_label_one_hot]+rbatch.data, []), is_train=False)
                outG = modG.get_outputs()

                modD.forward(mx.io.DataBatch(outG+[batch_label_one_hot], []), is_train=False)

                visual('gout', outG[0].asnumpy())
                visual('data', batch.data[0].asnumpy())

                if t % 100 == 0:
                    filename = "imgs/epoch-%d-batch-%d.jpg" % (epoch+1, t)
                    buff = convert2img(outG[0].asnumpy())
                    cv2.imwrite(filename, buff)

        if check_point:
            print 'Saving...'
            modG.save_params('%s_G_%s-%04d.params'%(dataset, stamp, epoch))
            modD.save_params('%s_D_%s-%04d.params'%(dataset, stamp, epoch))

