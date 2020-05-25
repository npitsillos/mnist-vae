# mnist-vae

A Variational Autoencoder implementation on MNIST images using convolutional layers.

Uses a training wrapper I created which can be found [here](https://github.com/npitsillos/productivity_efficiency/tree/master/torch_trainer).

Just clone both repos in your Desktop and run ```main.py```.

## Dataset Extension

The original MNIST dataset from pytorch was changed and a loss function was employed to be passed to the trainer wrapper.  The dataset simply extends the ```__getitem__()``` method to return the image as the target.

The trainer's loss function field is agnostic that is why a method was used.

## Latent Space Visualisation

<iframe id="igraph" scrolling="no" style="border:none; " seamless="seamless" src="./raw.html" height="525" width="100%"></iframe>