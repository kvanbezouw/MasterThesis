import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
powers_norm = [0.00523236, 0.01189665, 0.02028677, 0.03075027, 0.04369745, 0.05961211, 0.07906394, 0.10272296, 0.13137636, 0.16594795, 0.20752085, 0.25736381, 0.31696167, 0.38805081, 0.47266015, 0.57315869, 0.69231061, 0.83333906, 1, 0.00523236]
powers = [3.3 * i for i in powers_norm]
data_points = 20
df = pd.DataFrame()
data = np.empty(shape = [20, 9000, 7])
for i in range(0,data_points):
    data[i] = pd.read_csv(r"Z:\People\KvB\05-02\x_microm_disk\power_meas_1521,75nm{}".format(i),delimiter = ',')
data_t = np.transpose(data)

#Change MATLAB to python from here on.
# 
x_dat_temp = []
y_dat_temp = []
x_dat = []
y_dat = []
print(data.size())
print(data_t.size())
plt.figure(1)
for j in range(0,data_points):

    #print(data_t[j][0:5])
    #print(" ")
   
    x_dat_temp.append(data[j][3])
    y_dat_temp.append(data[j][4])

    x_dat.append(x_dat_temp[j][:6000]) #%NaNs after here
    y_dat.append(y_dat_temp[j][:6000]) #NaNs after here
  

    plt.plot(x_dat[j], y_dat[j])

x_dat = np.asarray(x_dat)
y_dat = np.asarray(y_dat)
plt.show()