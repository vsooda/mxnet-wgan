# mxnet-wgan
mxnet implement for Conditional Wasserstein GAN

You only need to change the `is_wgan` flag to test wgan or dcgan results.

note: 

* because we want to try mlp result, so I flatten the input image to a vector, then append condition one-hot vector to the vector. If you only want to try Convolution ops, you will not need to add condition in this way
* wgan seems not better than dcgan


# Acknowledgments

Code borrows from [mxnet gan example](https://github.com/dmlc/mxnet/blob/master/example/gan/dcgan.py),  [WGAN](https://github.com/luoyetx/WGAN)
