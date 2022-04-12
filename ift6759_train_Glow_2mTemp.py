import numpy as np
import xarray as xr
import tensorflow as tf
import datetime
import wandb
import os
import time

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
import matplotlib.pyplot as plt


def upsample(new_wt, new_ht, method, scale_factor=1):
    @tf.function
    def _upsample(x):
        return tf.image.resize(x, (new_wt,new_ht), method=method) / scale_factor
    return _upsample

def preprocess_vds(data_lo, data_hi, batch_size=100, buffer_size=1000, supervised=True, shuffle=True):
    if not shuffle:
        data = tf.data.Dataset.zip((data_lo, data_hi))
    else:
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

#Function for visualizing our samples
def plot_1xn(data, titles, cmin=-10., cmax=10.):
    n = len(data)
    plt.figure(figsize=(n*9,6))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(data[i].numpy().squeeze(), origin='lower')
        plt.colorbar(pad=0.04, shrink=0.5)
    plt.suptitle(titles, y=0.85)

class DataLoaderTemp:
    def __init__(
        self,
        path_lr='data/processed/temp/5625/temp_5625_processed.zarr',
        path_hr='data/processed/temp/1406/temp_1406_processed.zarr',
        n_total=None,
        batch_size=10
    ):
        # load the data
        zarr_lr = xr.open_zarr(path_lr)
        zarr_hr = xr.open_zarr(path_hr)

        # center it to zero
        zarr_lr, monthly_means_lr = remove_monthly_means(zarr_lr, time_dim='date')
        zarr_hr, monthly_means_hr = remove_monthly_means(zarr_hr, time_dim='date')

        # train and test split the zarr arrays
        assert len(zarr_hr.date)==len(zarr_lr.date)

        if n_total is None: n_total = len(zarr_hr.date)
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
        #import ipdb; ipdb.set_trace()

        # zipping the data together and shuffling each dataset individually
        # for "unsupervised learning"
        train_ds = preprocess_vds(dataset_train_lr, dataset_train_hr, batch_size=batch_size, buffer_size=n_train, supervised=False)
        valid_ds = preprocess_vds(dataset_valid_lr, dataset_valid_hr, batch_size=batch_size, buffer_size=n_valid, supervised=False)
        valid_ds_paired = preprocess_vds(dataset_valid_lr, dataset_valid_hr, batch_size=100, buffer_size=n_valid, supervised=True, shuffle=False)
        test_ds = preprocess_vds(dataset_test_lr, dataset_test_hr, batch_size=batch_size, buffer_size=n_test, supervised=False)
        test_ds_paired = preprocess_vds(dataset_test_lr, dataset_test_hr, batch_size=100, buffer_size=n_test, supervised=True, shuffle=False)

        scale = ndarray_hr_train.shape[1] // ndarray_lr_train.shape[1]

        # Setting attributes
        self.zarr_hr_valid = zarr_hr_valid
        self.zarr_hr_test = zarr_hr_test
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.valid_ds_paired = valid_ds_paired
        self.test_ds = test_ds
        self.test_ds_paired = test_ds_paired
        self.n_total = n_total
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.lon_hr = lon_hr
        self.lat_hr = lat_hr
        self.scale = scale
        self.monthly_means_lr = monthly_means_lr
        self.monthly_means_hr = monthly_means_hr



def main():
    # launching wandb
    wandb.init(project="Train-Glow-2mTemp")

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

    wandb.config.update({'validate_freq':validate_freq,
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
                    'n_epochs':n_epochs})

    dl = DataLoaderTemp(batch_size=sample_batch_size)

    flow_hr = Invert(GlowFlow(num_layers=layers, depth=depth, coupling_nn_ctor=coupling_nn_glow(max_filters=max_filters), name='glow_hr'))
    flow_lr = Invert(GlowFlow(num_layers=layers, depth=depth, coupling_nn_ctor=coupling_nn_glow(max_filters=max_filters), name='glow_lr'))


    dx = adversarial.PatchDiscriminator((dl.lat_hr, dl.lon_hr,1))
    dy = adversarial.PatchDiscriminator((dl.lat_hr, dl.lon_hr,1))
    model_joint = JointFlowLVM(flow_lr, flow_hr, dx, dy,
                                Gx_aux_loss=spatial_mae(dl.scale, stride=dl.scale),
                                Gy_aux_loss=spatial_mae(dl.scale),
                                input_shape=(None, dl.lat_hr, dl.lon_hr, 1))
    
    start_time = time.time()
    for i in range(n_epochs):
        # training
        print(f'Training joint model for {validate_freq} epochs ({i}/{n_epochs} complete)', flush=True)
        #model_joint.load_weights('model_checkpoints/test_jflvm_checkpoint')
        train_metrics = model_joint.train(dl.train_ds, steps_per_epoch=dl.n_train//sample_batch_size, num_epochs=validate_freq,lam=lam-lam_decay*validate_freq*i, lam_decay=lam_decay, alpha=alpha)

        # evaluation
        valid_eval_metrics = model_joint.evaluate(dl.valid_ds, dl.n_valid//sample_batch_size)
        
        # Sampling and Visualizing for every 3 epochs
        if i%2==0 and i!=0:
            #Sampling and Visualizing x and y
            samples_x,samples_y = model_joint.sample(n=4)  
            plot_1xn(samples_x, r"Samples $x \sim P(X)$")
            plt.savefig('sampling_figures/Unconditional_X_epoch{0:02d}'.format(i))
            plt.clf()
            plot_1xn(samples_y, r"Samples $y \sim P(Y)$")
            plt.savefig('sampling_figures/Unconditional_Y_epoch{0:02d}'.format(i))
            plt.clf()
            x_t, y_t = next(dl.test_ds_paired.__iter__())
            
            # Conditional Sampling
            xp_t = model_joint.predict_x(y_t)                
            yp_t = model_joint.predict_y(x_t)
            # Visualizing Inputs & Outputs
            plot_1xn([x_t[0], y_t[0], xp_t[0], yp_t[0]], r"Predictions $X \leftrightarrow Y$")
            plt.savefig('sampling_figures/Conditional_epoch{0:02d}'.format(i))
            plt.clf()
        
        # # Saving the model
        # model_joint.save(f'model_checkpoints/jflvm_checkpoint')
        
        # climdex
        print('Evaluating valid set ClimDEX indices on predictions')
        y_true, y_pred = [], []
        for x, y in dl.valid_ds_paired:
            y_true.append(y)
            z, ildj = model_joint.G_zx.inverse(x)
            y_, fldj = model_joint.G_zy.forward(z)
            y_pred.append(y_)
        y_true = tf.concat(y_true, axis=0)
        y_pred = tf.concat(y_pred, axis=0)

        #valid_mse = (tf.keras.metrics.mean_squared_error(y_true, y_pred))
        mse = tf.keras.losses.MeanSquaredError()
        valid_mse = mse(y_true,y_pred).numpy()
        #print(valid_mse)
        # computing climdex indices
        valid_txx_bias, valid_txn_bias = eval_climdex(np.squeeze(y_true.numpy()), np.squeeze(y_pred.numpy()), dl.zarr_hr_valid.coords)
        valid_txx_bias_mean, valid_txx_bias_std = valid_txx_bias.mean().values, valid_txx_bias.std().values
        valid_txn_bias_mean, valid_txn_bias_std = valid_txn_bias.mean().values, valid_txn_bias.std().values
        print('valid_mse:'+str(valid_mse))
        # printing climdex indices
        print('valid txx_bias_mean, valid txx_bias_std:', valid_txx_bias_mean, valid_txx_bias_std)
        print('valid txn_bias_mean, valid txn_bias_std:', valid_txn_bias_mean, valid_txn_bias_std)

        # logging losses, metrics in WandB
        for key, value in train_metrics.items():
            wandb.log({'train_'+key: value[0]}, step=i)
        for key, value in valid_eval_metrics.items():
            wandb.log({'valid_eval_'+key: value[0]}, step=i)
        # logging climdex indices in WandB
        wandb.log({'valid_mse':valid_mse}, step=i)
        wandb.log({'valid_txx_bias_mean':valid_txx_bias_mean}, step=i)
        wandb.log({'valid_txx_bias_std':valid_txx_bias_std}, step=i)
        wandb.log({'valid_txn_bias_mean':valid_txn_bias_mean}, step=i)
        wandb.log({'valid_txn_bias_std':valid_txn_bias_std}, step=i)
        
    #Test set evaluation
    test_eval_metrics = model_joint.evaluate(dl.test_ds, dl.n_test//sample_batch_size)

    print('Evaluating Test ClimDEX indices on predictions')
    y_true, y_pred = [], []
    for x, y in dl.test_ds_paired:
        y_true.append(y)
        z, ildj = model_joint.G_zx.inverse(x)
        y_, fldj = model_joint.G_zy.forward(z)
        y_pred.append(y_)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)

    # saving test y_pred and y_true as backup
    np.save('test_y_true.npy', y_true.numpy())
    np.save('test_y_pred.npy', y_pred.numpy())

    # computing climdex indices
    test_txx_bias, test_txn_bias = eval_climdex(np.squeeze(y_true.numpy()), np.squeeze(y_pred.numpy()), dl.zarr_hr_test.coords)
    test_txx_bias_mean, test_txx_bias_std = test_txx_bias.mean().values, test_txx_bias.std().values
    test_txn_bias_mean, test_txn_bias_std = test_txn_bias.mean().values, test_txn_bias.std().values

    # printing climdex indices
    print('test_txx_bias_mean, test_txx_bias_std:', test_txx_bias_mean, test_txx_bias_std)
    print('test_txn_bias_mean, test_txn_bias_std:', test_txn_bias_mean, test_txn_bias_std)

    total_training_time = time.time() - start_time
    print('total training time:', total_training_time)

    # logging losses, metrics in WandB
    for key, value in test_eval_metrics.items():
        wandb.log({'test_eval_'+key: value[0]}, step=i)
        # logging climdex indices in WandB
    wandb.log({'test_txx_bias_mean':test_txx_bias_mean}, step=i)
    wandb.log({'test_txx_bias_std':test_txx_bias_std}, step=i)
    wandb.log({'test_txn_bias_mean':test_txn_bias_mean}, step=i)
    wandb.log({'test_txn_bias_std':test_txn_bias_std}, step=i)
    wandb.log({'total_training_time':total_training_time}, step=i)

    # Saving the last model
    model_joint.save(f'model_checkpoints/final_jflvm_checkpoint')


if __name__ == "__main__":
    main()