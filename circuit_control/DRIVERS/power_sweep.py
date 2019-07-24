import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
import math
import numpy as np
import time
#create linear array of powers, from P = P_0 to 23 db attenuation of P_0.
from DRIVERS.yellow_cDAQ import cDAQtest
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class Power_Sweep():
    #If this class is inherited, then so are the class variables

    v_voa_calib_points = np.asarray(np.linspace(0,4.1,42)) 
    P_vvoa1 = np.asarray([1472, 1468, 1462, 1453, 1441, 1426, 1406, 1382, 1353, 1320, 1281, 1237, 1188, 1134, 1076, 1014, 947, 878, 807, 735, 662, 590, 520, 452, 388, 329, 275, 225.5, 182, 144.3, 112.3, 85.8, 63, 46.90, 33.45, 23.0, 15.75, 10.38, 6.71, 4.230, 2.558, 1.558])/1472
    P_vvoa2 = np.asarray([1320, 1317, 1309, 1301, 1291, 1278, 1262, 1242, 1222, 1197, 1167, 1136, 1101, 1062, 1020, 974, 925, 875, 821, 765, 708, 648, 591, 533, 475, 420, 367, 317, 270, 227, 188, 153, 123, 97, 75, 57, 42, 30, 21.7, 14.96, 10.03,6.55])/1320
    P_vvoa1_run2 = np.asarray([68, 56.8, 47.18, 38.67, 31.77, 25.90, 21.09, 16.86, 13.48, 10.80, 8.61, 6.70, 5.28, 4.096]) #From 3.7 to 4.22 volts in steps of 0.04
    P_vvoa2_run2 = np.asarray([68.5, 59.9, 52, 45.09, 38.91, 33.50, 28.63, 24.42, 20.65, 17.48, 14.75, 12.29, 10.30, 8.53, 7.07, 5.88, 4.807, 3.804, 3.234, 2.641, 2.163]) #From 3.7 to 4.5 volts in steps of 0.04
    v_voa1_run2_calib_points = np.asarray(np.linspace(3.7,4.22,14))
    v_voa2_run2_calib_points = np.asarray(np.linspace(3.7,4.5,21))
    P_vvoa1_run2 = P_vvoa1_run2/P_vvoa1_run2[1] * P_vvoa1[37] #calibrate wrt the value at 3.7 V 
    P_vvoa2_run2 = P_vvoa2_run2/P_vvoa2_run2[1] * P_vvoa2[37] #calibrate wrt the value at 3.7 V

    v_voa1_calibpoints_tot = np.concatenate((v_voa_calib_points[0:37], v_voa1_run2_calib_points))
    v_voa2_calibpoints_tot = np.concatenate((v_voa_calib_points[0:37], v_voa2_run2_calib_points))

    P_vvoa1_tot = np.concatenate((P_vvoa1[0:37], P_vvoa1_run2))
    P_vvoa2_tot = np.concatenate((P_vvoa2[0:37], P_vvoa2_run2))

    VOA1_attenuation = 10*np.asarray(np.log10(P_vvoa1_tot))
    VOA2_attenuation = 10*np.asarray(np.log10(P_vvoa2_tot))
   
    volt_vs_atten_voa1 = interp1d(VOA1_attenuation, v_voa1_calibpoints_tot, kind = 'cubic')
    volt_vs_atten_voa2 = interp1d(VOA2_attenuation, v_voa2_calibpoints_tot, kind = 'cubic') #cubic gives slightly better results than linear interpol.
 
    
    def __init__(self):
        #Calibration data. VOAs calibrated up til V  = 4.1. Not further because of psosible instabilities/inconsistencies etc.
        pass
        

        
    def set_voas(self, max_attenuation, attenuation_voa1, channel_voa1, channel_voa2):
        channel_voa1 = str(channel_voa1)
        channel_voa2 = str(channel_voa2)
        try:
            print("Max atten {}".format(max_attenuation))
            print("Atten voa 1 {}".format(attenuation_voa1))
            
            assert max_attenuation <= float(0) 
            assert max_attenuation >= float(-42)
            assert attenuation_voa1 <= 0
            assert channel_voa1 != channel_voa2
            attenuation_voa2 = max_attenuation - attenuation_voa1
            print("Atten voa 2 {}".format(attenuation_voa2))
            assert attenuation_voa2 <= 0
            v_voa1 = self.volt_vs_atten_voa1(attenuation_voa1)
            v_voa2 = self.volt_vs_atten_voa2(attenuation_voa2)
            
            #prevent overloading: first set the smaller attenuation to 5V, then induce the largest attenuation, then go from 5V to the smaller attenuation
            if abs(v_voa2) >= abs(v_voa1):
                cDAQtest.cDAQ_write_DC(channel = channel_voa1, voltage = 5)
                time.sleep(0.05)
                cDAQtest.cDAQ_write_DC(channel = channel_voa2, voltage = v_voa2)
                time.sleep(0.05)
                cDAQtest.cDAQ_write_DC(channel = channel_voa1, voltage = v_voa1)
            else:
                cDAQtest.cDAQ_write_DC(channel = channel_voa2, voltage = 5)
                time.sleep(0.05)
                cDAQtest.cDAQ_write_DC(channel = channel_voa1, voltage = v_voa1)
                time.sleep(0.05)
                cDAQtest.cDAQ_write_DC(channel = channel_voa2, voltage = v_voa2)
            
            return v_voa1, v_voa2
        except AssertionError:
            print("Your voas are set incorrectly. Look in power_sweep.py to check the possible assertion error causes")
            raise AssertionError

    @staticmethod
    def create_non_linear_attenuation_vector(data_points, density_decrease = 2.85, scaling = 1, max_power = 3.3):
        try:
            scaling = float(scaling)
            assert data_points > 0
            assert data_points > 0
            assert scaling <= 1
         

            p_vec = np.linspace(0, scaling, data_points)
            p_vec_new_scaling = [p_vec[i]*density_decrease**(i/data_points) for i in range(0, data_points)] #density at low powers is approx dens_decr times density at high powers
            p_vec_new_scaling = p_vec_new_scaling/p_vec_new_scaling[-1] * scaling
            
            p_vec_new_scaling = p_vec_new_scaling - p_vec_new_scaling[1] + 0.0008 #keep the first datapoint at this number * 3.3  always, to keep the same amplification etc.

            p_vec_new_scaling = np.delete(p_vec_new_scaling, 0) 
            p_vec_new_scaling = np.append(p_vec_new_scaling,p_vec_new_scaling[0])
    
            atten_vec = 10 * np.log10(p_vec_new_scaling)
            p_vec = p_vec * max_power #scale with maximum power ( = 0 atten)
            #print(atten_vec)

            return p_vec_new_scaling, atten_vec
        except AssertionError:
            print("Your attenuation vector has wrong input data")
            raise AssertionError
        
    @staticmethod   #first creates linear array of values, then converts to dB
    def create_attenuation_vector(max_attenuation, data_points, scaling, max_power = 3.3):
        try:
            max_power = float(max_power)
            assert data_points > 0
            assert max_attenuation < 0
            assert max_power <= 1
            p_vec = np.linspace(10**(max_attenuation/10), scaling * max_power, data_points)
            atten_vec = 10*np.log10(p_vec)

            return p_vec, atten_vec
        except AssertionError:
            print("Your attenuation vector has wrong input data")
            raise AssertionError

            
if __name__ == '__main__':
    print("This is a helper class to keep the fine scan code tidy, it's not meant to be ran mainn")
    #a, b = Power_Sweep.create_non_linear_attenuation_vector(data_points = 20, density_decrease = 4, scaling = 0.6, max_power = 3.3)
   
    pass