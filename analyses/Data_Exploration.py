
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util import load_data
import paths


mode = 'SeaGL'
preload_IO = True
TCCON = True
qf = None

data_list = []
for year in range(2015, 2023):
    data = load_data(year, mode, qf=qf, preload_IO=preload_IO, TCCON=TCCON)
    data_list.append(data)
data = pd.concat(data_list)

# compare to TCCON
data['xco2-TCCON'] = data['xco2_raw'] - data['xco2tccon']
tccon_median = data.groupby('tccon_name').mean()


# fit linear to data

y = tccon_median['xco2-TCCON'].to_numpy()
x = tccon_median['sensor_zenith_angle'].to_numpy()

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)


# plot data and model
plt.figure(figsize=(10, 7))
plt.scatter(x, y, marker='x')
plt.show()



for var in data.columns:
    try:
        plt.figure(figsize=(10, 7))
        plt.scatter(tccon_median[var], tccon_median['xco2-TCCON'], marker='x')
        for i, txt in enumerate(tccon_median.index):
            plt.annotate(txt, (tccon_median[var][i], tccon_median['xco2-TCCON'][i] + 0.005))
        plt.ylabel('OCO-2 bias by TCCON Station')
        plt.xlabel(var)
        h_max = np.nanmax(tccon_median[var])
        h_min = np.nanmin(tccon_median[var])
        plt.hlines(0, h_min, h_max, colors='k', linestyles='dashed')
        plt.title('OCO-2_raw - TCCON [ppm]')
        plt.tight_layout()
        plt.savefig(paths.PLOTS_DIR / 'TCCON' / 'OCO2raw-TCCON_vs_'+var+'.png')
        # plt.show()
        plt.close()
    except:
        continue
