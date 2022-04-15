from ift6759_train_Glow_2mTemp import DataLoaderTemp
from baselines.bcsd import BCSD
from tqdm import tqdm
import numpy as np
import xarray as xr

def main():
    dl = DataLoaderTemp()

    bcsd = BCSD(verbose=True, time_dim="date")
    bcsd.fit(dl.zarr_lr_train['t2m'], dl.zarr_hr_train['t2m'])

    n_samples = dl.zarr_lr_test.dims['date']
    print('Computing {n_samples} predictions...')
    preds = []
    for i in range(int(n_samples/100)+1):
        if i == int(n_samples/100):
            preds.append(bcsd.predict(dl.zarr_lr_test['t2m'][i*100:]).compute())
        else:
            preds.append(bcsd.predict(dl.zarr_lr_test['t2m'][i*100:(i+1)*100]).compute())

    # concatenating the preds xr.dataArrays
    pred_hr_test = xr.concat(preds, dim='date')

    # Compute metrics
    true = dl.zarr_hr_test['t2m'].to_numpy()
    pred = pred_hr_test.to_numpy()

    # correlation coefficients
    corrs=[]
    for i in tqdm(range(len(pred)), 'CorrCoef'):
        corrs.append(np.corrcoef(pred[i].flatten(), true[i].flatten())[0,1])
    print(f'corrs: {np.mean(corrs)} +/- {np.std(corrs)}' )

    # mean square root error
    mse=[]
    for i in tqdm(range(len(pred)), 'MSE'):
        mse.append(
            np.mean((pred[i] - true[i])**2)
        )
    print(f'mse: {np.mean(mse)} +/- {np.std(mse)}' )

    pred_hr_test.to_zarr('BCSD_pred_hr_test.zarr')
    dl.zarr_hr_test['t2m'].to_zarr('BCSD_true_hr_test.zarr')
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()
