import numpy as np
import xarray as xr
import tensorflow as tf

def upsample(new_wt, new_ht, method, scale_factor=1):
    @tf.function
    def _upsample(x):
        return tf.image.resize(x, (new_wt,new_ht), method=method) / scale_factor
    return _upsample

# load the data
zarr_lr = xr.open_zarr('data/processed/temp/5625/temp_5625_processed.zarr')
zarr_hr = xr.open_zarr('data/processed/temp/1406/temp_1406_processed.zarr')

# make data numpy arrays (unsuited for large data):
# each ndarray have shape [date, lat, lon]
ndarray_lr = zarr_lr.to_array().to_numpy().squeeze()
ndarray_hr = zarr_hr.to_array().to_numpy().squeeze()

# defining tensorflow datasets
# this batches the data along the date axis to yield
# 1096 samples of shape [lat, lon]
# We keep the last eighth of the sample for test set.
assert len(ndarray_lr)==len(ndarray_hr)
test_size = int(1/8 * len(ndarray_lr))

dataset_train_lr = tf.data.Dataset.from_tensor_slices(ndarray_lr[:-test_size])
dataset_train_hr = tf.data.Dataset.from_tensor_slices(ndarray_hr[:-test_size])
dataset_test_lr = tf.data.Dataset.from_tensor_slices(ndarray_lr[-test_size:])
dataset_test_hr = tf.data.Dataset.from_tensor_slices(ndarray_hr[-test_size:])

# We need to naively upsample the low res data so it has the same
# dimensionality as the high res using the nearest neighbour algo
# TODO 
# dataset_train_lr = dataset_train_lr.map(upsample(wt_hi, ht_hi, tf.image.ResizeMethod.NEAREST_NEIGHBOR))

import ipdb;ipdb.set_trace()

# zipping the data together and shuffling each dataset individually
# for "unsupervised learning"
buffer_size=1000
data = tf.data.Dataset.zip((dataset_train_lr.shuffle(buffer_size), dataset_train_hr.shuffle(buffer_size)))




