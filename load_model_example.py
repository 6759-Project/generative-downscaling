import tensorflow as tf
from ift6759_train_Glow_2mTemp import DataLoaderTemp
from normalizing_flows.models import VariationalModel, FlowLVM, JointFlowLVM, adversarial
from normalizing_flows.flows import Invert
from normalizing_flows.flows.glow import GlowFlow, coupling_nn_glow

def spatial_mae(scale, stride=1):
    """
    "Spatial" MAE auxiliary loss for generator. Penalizes outputs
    which violate spatial average preservation between input and output.
    """
    kernel = tf.ones((scale,scale,1,1)) / (scale**2.)
    def _spatial_mse(x_in, y_pred):
        x_avg = tf.nn.conv2d(x_in, kernel, strides=(stride, stride), padding='VALID')
        y_avg = tf.nn.conv2d(y_pred, kernel, strides=(stride, stride), padding='VALID')
        return tf.math.reduce_mean(tf.math.abs(y_avg - x_avg))
    return _spatial_mse

# dataset
dl = DataLoaderTemp(batch_size=10)

# creating the model (it needs to be the same as the one you're trying to load! this should work for the final model)
flow_hr = Invert(GlowFlow(num_layers=4, depth=8, coupling_nn_ctor=coupling_nn_glow(max_filters=256), name='glow_hr'))
flow_lr = Invert(GlowFlow(num_layers=4, depth=8, coupling_nn_ctor=coupling_nn_glow(max_filters=256), name='glow_lr'))

dx = adversarial.PatchDiscriminator((dl.lat_hr, dl.lon_hr,1))
dy = adversarial.PatchDiscriminator((dl.lat_hr, dl.lon_hr,1))
model_joint = JointFlowLVM(flow_lr, flow_hr, dx, dy,
                            Gx_aux_loss=spatial_mae(dl.scale, stride=dl.scale),
                            Gy_aux_loss=spatial_mae(dl.scale),
                            input_shape=(None, dl.lat_hr, dl.lon_hr, 1))


model_joint.load('model_checkpoints/test_jflvm_checkpoint', 2)