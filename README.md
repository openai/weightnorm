
# Weight Normalization

This repo contains example code for [Weight Normalization](https://arxiv.org/abs/1602.07868), as described in the following 
paper:

**Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks**, by
Tim Salimans, and Diederik P. Kingma.

- The folder 'lasagne' contains code using the Lasagne package for Theano. This code was used to run the CIFAR-10 experiments in the paper.
- The folder 'tensorflow' contains a single nn.py file with a direct implementation copied from our [PixelCNN++](https://github.com/openai/pixel-cnn) repository.
- The folder 'keras' contains example code for use with the Keras package.

## Citation

If you find this code useful please cite us in your work:

```
@inproceedings{Salimans2016WeightNorm,
  title={Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks},
  author={Tim Salimans and Diederik P. Kingma},
  booktitle={Neural Information Processing Systems 2016},
  year={2016}
}
```
