"""Plot actual synthetic-image posterior checks; label their limited scope."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path('.cache/bayes3d-pose-probe')
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,label in zip(axes,('far','near')):
    with np.load(root/f'{label}_0.005.npz') as data:
        poses,weights=data['particles'],data['weights']
        yaw,x,y=[np.unique(poses[:,i]) for i in range(3)]
        marginal=weights.reshape(len(yaw),len(x),len(y)).sum(axis=0)
        im=ax.pcolormesh(x*1000,y*1000,marginal.T,shading='auto')
        ax.plot(12,-9,'rx',label='Synthetic generator truth (test only)')
        ax.set(xlabel='x (mm)',ylabel='y (mm)',title=f'{label.capitalize()} view: posterior position mass')
        fig.colorbar(im,ax=ax,label='Probability per grid cell')
        ax.legend(fontsize=7)
fig.suptitle('Bayes3D adapter synthetic-image check; no robot or planner result')
fig.savefig(root/'posteriors.png',dpi=160)
print(root/'posteriors.png')
