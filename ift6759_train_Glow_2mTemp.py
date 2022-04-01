import numpy as np
import xarray as xr
import tensorflow as tf
import datetime
import wandb
import os

# to install climdex:
# python -m pip install git+https://github.com/bgroenks96/pyclimdex.git

# some how this doesn't work:
# python -m pip install git+https://github.com/bgroenks96/normalizing-flows

# I copied the normalizing_flow folder from github (same commit as the
# submodule in Groenke's generative-downscaling points) in this project.
# I also had to downgrade from tensorflow-probability==0.16.0 to 0.15.0
# reminder to generate new environment.yml if all works out


from utils.preprocessing import remove_monthly_means
import climdex.temperature as tdex
from normalizing_flows.models import VariationalModel, FlowLVM, JointFlowLVM, adversarial
from normalizing_flows.models.variational import nll_loss
from normalizing_flows.models.optimization import LinearWarmupSchedule
from normalizing_flows.flows import Transform, Flow, Invert
from normalizing_flows.flows.image import Upsample
from normalizing_flows.flows.glow import GlowFlow, coupling_nn_glow
from utils.distributions import normal
from tensorflow.keras.optimizers import Adamax

wandb.init(project="Train-Glow-2mTemp", entity="6759-proj")

def upsample(new_wt, new_ht, method, scale_factor=1):
    @tf.function
    def _upsample(x):
        return tf.image.resize(x, (new_wt,new_ht), method=method) / scale_factor
    return _upsample

def preprocess_vds(data_lo, data_hi, batch_size=100, buffer_size=1000, supervised=True):
    if supervised:
        data = tf.data.Dataset.zip((data_lo, data_hi)).shuffle(buffer_size)
    else:
        data = tf.data.Dataset.zip((data_lo.shuffle(buffer_size), data_hi.shuffle(buffer_size)))
    return data.batch(batch_size)

indices = tdex.indices('date')
def eval_climdex(true, pred, coords):
    true_arr = xr.DataArray(true, coords=coords)
    pred_arr = xr.DataArray(pred, coords=coords)
    txx_true = indices.monthly_txx(true_arr)
    txx_pred = indices.monthly_txx(pred_arr)
    txn_true = indices.monthly_txn(true_arr)
    txn_pred = indices.monthly_txn(pred_arr)
    txx_bias = txx_pred - txx_true
    txn_bias = txn_pred - txn_true
    return txx_bias, txn_bias

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


# load the data
zarr_lr = xr.open_zarr('data/processed/temp/5625/temp_5625_processed.zarr')
zarr_hr = xr.open_zarr('data/processed/temp/1406/temp_1406_processed.zarr')

# center it to zero
zarr_lr, monthly_means_lr = remove_monthly_means(zarr_lr, time_dim='date')
zarr_hr, monthly_means_hr = remove_monthly_means(zarr_hr, time_dim='date')

# train and test split the zarr arrays
assert len(zarr_hr.date)==len(zarr_lr.date)

n_total = len(zarr_hr.date) # uncomment this to run on full dataset
n_train = int(0.7*n_total)
n_valid = int(0.2*n_total)
n_test = n_total-n_train-n_valid


zarr_lr_train = zarr_lr.isel(date=slice(0, n_train))
zarr_hr_train = zarr_hr.isel(date=slice(0, n_train))

zarr_lr_valid = zarr_lr.isel(date=slice(n_train, n_train+n_valid))
zarr_hr_valid = zarr_hr.isel(date=slice(n_train, n_train+n_valid))

zarr_lr_test = zarr_lr.isel(date=slice(n_train+n_valid, n_train+n_valid+n_test))
zarr_hr_test = zarr_hr.isel(date=slice(n_train+n_valid, n_train+n_valid+n_test))

# make data numpy arrays (unsuited for large data):
# each ndarray have shape [date, lat, lon]
ndarray_lr_train = zarr_lr_train.to_array().to_numpy().squeeze()
ndarray_hr_train = zarr_hr_train.to_array().to_numpy().squeeze()

ndarray_lr_valid = zarr_lr_valid.to_array().to_numpy().squeeze()
ndarray_hr_valid = zarr_hr_valid.to_array().to_numpy().squeeze()

ndarray_lr_test = zarr_lr_test.to_array().to_numpy().squeeze()
ndarray_hr_test = zarr_hr_test.to_array().to_numpy().squeeze()

# defining tensorflow datasets
# this batches the data along the date axis to yield
# n_total samples of shape [lat, lon]
# We keep the last eighth of the sample for test set.
dataset_train_lr = tf.data.Dataset.from_tensor_slices(ndarray_lr_train)
dataset_train_hr = tf.data.Dataset.from_tensor_slices(ndarray_hr_train)

dataset_valid_lr = tf.data.Dataset.from_tensor_slices(ndarray_lr_valid)
dataset_valid_hr = tf.data.Dataset.from_tensor_slices(ndarray_hr_valid)

dataset_test_lr = tf.data.Dataset.from_tensor_slices(ndarray_lr_test)
dataset_test_hr = tf.data.Dataset.from_tensor_slices(ndarray_hr_test)

# We need to naively upsample the low res data so it has the same
# dimensionality as the high res using the nearest neighbour algo
# We first add a channel to all images
dataset_train_lr = dataset_train_lr.map(lambda x: x[:,:,None])
dataset_train_hr = dataset_train_hr.map(lambda x: x[:,:,None])

dataset_valid_lr = dataset_valid_lr.map(lambda x: x[:,:,None])
dataset_valid_hr = dataset_valid_hr.map(lambda x: x[:,:,None])

dataset_test_lr = dataset_test_lr.map(lambda x: x[:,:,None])
dataset_test_hr = dataset_test_hr.map(lambda x: x[:,:,None])

# import ipdb;ipdb.set_trace()
# Then upsample the low res datasets
lat_hr, lon_hr = ndarray_hr_train.shape[1:]
dataset_train_lr = dataset_train_lr.map(upsample(lat_hr, lon_hr, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
dataset_valid_lr = dataset_valid_lr.map(upsample(lat_hr, lon_hr, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
dataset_test_lr = dataset_test_lr.map(upsample(lat_hr, lon_hr, tf.image.ResizeMethod.NEAREST_NEIGHBOR))
import ipdb; ipdb.set_trace()

#hemanth continue here

# zipping the data together and shuffling each dataset individually
# for "unsupervised learning"
train_ds = preprocess_vds(dataset_train_lr, dataset_train_hr, batch_size=10, buffer_size=n_train, supervised=False)
test_ds = preprocess_vds(dataset_test_lr, dataset_test_hr, batch_size=10, buffer_size=n_test, supervised=False)

flow_hr = Invert(GlowFlow(num_layers=3, depth=8, coupling_nn_ctor=coupling_nn_glow(max_filters=256), name='glow_hr'))
flow_lr = Invert(GlowFlow(num_layers=3, depth=8, coupling_nn_ctor=coupling_nn_glow(max_filters=256), name='glow_lr'))

scale = ndarray_hr_train.shape[1] // ndarray_lr_train.shape[1]
dx = adversarial.PatchDiscriminator((lat_hr, lon_hr,1))
dy = adversarial.PatchDiscriminator((lat_hr, lon_hr,1))
model_joint = JointFlowLVM(flow_lr, flow_hr, dx, dy,
                            Gx_aux_loss=spatial_mae(scale, stride=scale),
                            Gy_aux_loss=spatial_mae(scale),
                            input_shape=(None,lat_hr, lon_hr,1))

# these are all args in his fit_glow_jflvm() function from glow-downscaling-maxt.ipynb
validate_freq=1
warmup=1
sample_batch_size=10
load_batch_size=1200
layers=4
depth=8
min_filters=32
max_filters=256
lam=1.0
lam_decay=0.01
alpha=1.0
n_epochs=20

wandb.config = {'validate_freq':validate_freq,
                'warmup':warmup,
                'sample_batch_size':sample_batch_size,
                'load_batch_size':load_batch_size,
                'layers':layers,
                'depth':depth,
                'min_filters':min_filters,
                'max_filters':max_filters,
                'lam':lam,
                'lam_decay':lam_decay,
                'alpha':alpha,
                'n_epochs':n_epochs}

for i in range(n_epochs):
    # training
    print(f'Training joint model for {validate_freq} epochs ({i}/{n_epochs} complete)', flush=True)
    train_metrics = model_joint.train(train_ds, steps_per_epoch=n_train//sample_batch_size, num_epochs=validate_freq,
                          lam=lam-lam_decay*validate_freq*i, lam_decay=lam_decay, alpha=alpha)

    # evaluation
    eval_metrics = model_joint.evaluate(test_ds, n_test//sample_batch_size)
    model_joint.save('/tmp/test_jflvm_checkpoint')

    # climdex
    print('Evaluating ClimDEX indices on predictions')
    y_true, y_pred = [], []
    for x, y in tf.data.Dataset.zip((dataset_test_lr, dataset_test_hr)).batch(2*sample_batch_size):
        y_true.append(y)
        z, ildj = model_joint.G_zx.inverse(x)
        y_, fldj = model_joint.G_zy.forward(z)
        y_pred.append(y_)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    # computing climdex indices
    txx_bias, txn_bias = eval_climdex(np.squeeze(y_true.numpy()), np.squeeze(y_pred.numpy()), zarr_hr_test.coords)
    txx_bias_mean, txx_bias_std = txx_bias.mean().values, txx_bias.std().values
    txn_bias_mean, txn_bias_std = txn_bias.mean().values, txn_bias.std().values

    # printing climdex indices
    print('txx_bias_mean, txx_bias_std:', txx_bias_mean, txx_bias_std)
    print('txn_bias_mean, txn_bias_std:', txn_bias_mean, txn_bias_std)

    # logging losses, metrics in WandB
    for key, value in train_metrics.items():
        wandb.log({'train_'+key: value[0]})
    for key, value in eval_metrics.items():
        wandb.log({'eval_'+key: value[0]})
    # logging climdex indices in WandB
    wandb.log({'txx_bias_mean':txx_bias_mean})
    wandb.log({'txx_bias_std':txx_bias_std})
    wandb.log({'txn_bias_mean':txn_bias_mean})
    wandb.log({'txn_bias_std':txn_bias_std})

# saving the model
if not os.path.exists('final_model'):
    os.mkdir('./final_model')
finale_model_path = model_joint.save('./final_model/')
print('final model saved at:', finale_model_path)