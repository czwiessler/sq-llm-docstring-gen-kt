import tensorflow as tf

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.vae import VariationalAutoencoder


class Model(tf.keras.models.Model):
    def __init__(self,
                 data_format='channels_last',
                 groups=8,
                 reduction=2,
                 l2_scale=1e-5,
                 dropout=0.2,
                 downsampling='conv',
                 upsampling='conv',
                 base_filters=16,
                 depth=4,
                 in_ch=2,
                 out_ch=3):
        """ Initializes the model, a cross between the 3D U-net
            and 2018 BraTS Challenge top model with VAE regularization.

            References:
                - [3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/pdf/1606.06650.pdf)
                - [3D MRI brain tumor segmentation using autoencoder regularization](https://arxiv.org/pdf/1810.11654.pdf)
        """
        super(Model, self).__init__()
        self.epoch = tf.Variable(0, name='epoch', trainable=False)
        self.encoder = Encoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            dropout=dropout,
                            downsampling=downsampling,
                            base_filters=base_filters,
                            depth=depth)
        self.decoder = Decoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            upsampling=upsampling,
                            base_filters=base_filters,
                            depth=depth,
                            out_ch=out_ch)
        self.vae = VariationalAutoencoder(
                            data_format=data_format,
                            groups=groups,
                            reduction=reduction,
                            l2_scale=l2_scale,
                            upsampling=upsampling,
                            base_filters=base_filters,
                            depth=depth,
                            out_ch=in_ch)

    def call(self, inputs, training=None, inference=None):
        # Inference mode does not evaluate VAE branch.
        assert (not inference or not training), \
            'Cannot run training and inference modes simultaneously.'

        inputs = self.encoder(inputs, training=training)

        y_pred = self.decoder((inputs[-1], inputs[:-1]), training=training)

        if inference:
            return (y_pred, None, None, None)
        y_vae, z_mean, z_logvar = self.vae(inputs[-1], training=training)

        return (y_pred, y_vae, z_mean, z_logvar)
