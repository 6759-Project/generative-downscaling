import numpy as np
import matplotlib.pyplot as plt

climdex_metrics = np.load('custom_logs/preliminary_results_climdex_03-19-2022_22-27-37.npy',allow_pickle=True)
metrics = np.load('custom_logs/preliminary_results_metrics_03-19-2022_22-27-37.npy',allow_pickle=True)
print(metrics[0].keys())
print(climdex_metrics[0].keys())

'''
print(len(climdex_metrics))
clim_metrics = climdex_metrics[0]

for i,metric in enumerate(climdex_metrics):
    if i>0:
        clim_metrics['txx_bias_mean']= np.append(clim_metrics['txx_bias_mean'] ,metric['txx_bias_mean'])
        clim_metrics['txx_bias_std']= np.append(clim_metrics['txx_bias_std'] ,metric['txx_bias_std'])
        clim_metrics['txn_bias_mean']= np.append(clim_metrics['txn_bias_mean'] ,metric['txn_bias_mean'])
        clim_metrics['txn_bias_std']= np.append(clim_metrics['txn_bias_std'] ,metric['txn_bias_std'])


plt.figure(figsize=(15,7))
plt.plot(range(0,6), clim_metrics['txx_bias_mean'],label='mean of txx bias')
plt.fill_between(range(0,6),clim_metrics['txx_bias_mean']-clim_metrics['txx_bias_std'],clim_metrics['txx_bias_mean']+clim_metrics['txx_bias_std'],alpha=0.5)
plt.axhline(y=0, color='blue', linestyle='--',label='no bias')

plt.ylim(-10,10)
plt.title('Epochs vs txx_bias',fontsize=14)
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('txx_bias',fontsize=14)
plt.legend()
plt.show()


plt.figure(figsize=(15,7))
plt.plot(range(0,6), clim_metrics['txn_bias_mean'],label='mean of txn bias')
plt.fill_between(range(0,6),clim_metrics['txn_bias_mean']-clim_metrics['txn_bias_std'],clim_metrics['txn_bias_mean']+clim_metrics['txn_bias_std'],alpha=0.5)
plt.axhline(y=0, color='blue', linestyle='--',label='no bias')
plt.ylim(-10,10)
plt.title('Epochs vs txn_bias',fontsize=14)
plt.xlabel('Epochs',fontsize=14)
plt.ylabel('txn_bias',fontsize=14)
plt.legend()
plt.show()
'''

metri = {}
for key in metrics[0].keys():
    metri[key] = []

for i,metric in enumerate(metrics):
    for key in metrics[i].keys():
        metri[key].append(metrics[i][key][0])

plt.plot(range(0,6),metri['nll_x'],label='nll_x')
plt.plot(range(0,6),metri['nll_y'],label='nll_y')
plt.title('Epochs vs NLL')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()

plt.plot(range(0,6),metri['gx_loss'],label='gx_loss')
plt.plot(range(0,6),metri['gy_loss'],label='gy_loss')
plt.title('Epochs vs g_loss')
plt.xlabel('Epochs')
plt.ylabel('g_loss')
plt.legend()
plt.show()

plt.plot(range(0,6),metri['dx_loss'],label='dx_loss')
plt.plot(range(0,6),metri['dy_loss'],label='dy_loss')
plt.title('Epochs vs d_loss')
plt.xlabel('Epochs')
plt.ylabel('d_loss')
plt.legend()
plt.show()


plt.plot(range(0,6),metri['gx_aux'],label='gx_aux')
plt.plot(range(0,6),metri['gy_aux'],label='gy_aux')
plt.title('Epochs vs g_aux')
plt.xlabel('Epochs')
plt.ylabel('g_aux')
plt.legend()
plt.show()






