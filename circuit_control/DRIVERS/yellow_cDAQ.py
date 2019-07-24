import numpy as np
import sys
from scipy import signal 
import nidaqmx
from nidaqmx import constants
from nidaqmx.constants import FuncGenType
import time
from nidaqmx.stream_readers import (AnalogSingleChannelReader, AnalogMultiChannelReader)
import PyDAQmx as nidaq
from nidaqmx.constants import Edge
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

#from PyQt5 import QtGui
#import time


class cDAQtest():
    def __init__(self):
     
        channelout = "cDAQ1Mod1/ao0"
        #x = self.cDAQ_read("cDAQ1Mod3/ai0", coupling = "DC")
        #print(x)
        #channel = task.ai_channels.add_ai_voltage_chan("cDAQ1Mod3/ai0") 
        #self.cDAQ_write_DC("cDAQ1Mod1/ao0", voltage = 5)
        #self.cDAQ_write_AC(channelout)
        #data = self.cDAQ_test()
    
    @staticmethod            #all static methods so that no instances have to be created 
    def cDAQ_read(channel):
        with nidaqmx.Task() as task:
            chan = task.ai_channels.add_ai_voltage_chan(channel)
            voltage = task.read()
        return voltage

    @staticmethod
    def cDAQ_read_cascade(channel, channel2 = None, coupling = "DC", Samples_Per_Sec = 1000):
        with nidaqmx.Task() as task:
            chan1 = task.ai_channels.add_ai_voltage_chan(channel)
            #chan2 = task.ai_channels.add_ai_voltage_chan(channel2)
            if (coupling == "DC"):
                chan1.ai_coupling = constants.Coupling.DC
            if (coupling == "AC"):                                  #AC coupling returns an error. https://knowledge.ni.com/KnowledgeArticleDetails?id=kA00Z0000019M4CSAU&l=nl-NL
                chan1.ai_coupling = constants.Coupling.AC
           
            Samples_Per_Ch_To_Read = 2000
            task.timing.cfg_samp_clk_timing(rate = Samples_Per_Sec)
            data = task.read(Samples_Per_Ch_To_Read)
           
            #task.timing.cfg_samp_clk_timing(Samples_Per_Sec, RISING,FINITE,Samples_Per_Ch_To_Read)    
        return data

    @staticmethod 
    def cDAQ_write_AC(channel):
         with nidaqmx.Task() as task:
            task.ao_channels.add_ao_func_gen_chan(channel, amplitude=5.0,offset=0.0) #AC again doesnt seem to work (the entirefunction doesn't work. Not needed for thesis anyway)..

    @staticmethod
    def cDAQ_write_DC(channel, voltage):
        voltage = float(voltage)
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(channel)
            task.write(voltage)
            #task.control(nidaqmx.constants.ExcitationDCorAC)
            #task.write(0)
            #task.write(1.1)
            #time.sleep(2)
            #task.write(2.2)
            #time.sleep(2)
            #task.write(3.2)
            #time.sleep(2)



if __name__ == '__main__':
    #print(nidaqmx.system.System.devices)  #check to see if devices are connected
    cDAQtest = cDAQtest()
    #cDAQtest.cDAQ_write_DC(channel = "cDAQ1Mod1/ao1", voltage = 0)
    #data = cDAQtest.cDAQ_read_cascade(channel = "cDAQ1Mod3/ai0", coupling = "DC", Samples_Per_Sec = 1000)
    #writetask = nidaqmx.task.Task()
    #writetask.ao_channels.add_ao_voltage_chan("cDAQ1Mod1/ao0", min_val = -1, max_val = 1)
    #V_voa = np.asarray([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    #P = np.asarray([115.7, 111.8, 99.7, 78.0, 50.45, 24.71, 8.31, 1.67, 0.1657, 0.00989, 0.00655])/115.7
    V_VOA_3 = np.asarray(np.linspace(0,4.1,42))
    print(V_VOA_3[0:5])
    P_vvoa2 = np.asarray([1320, 1317, 1309, 1301, 1291, 1278, 1262, 1242, 1222, 1197, 1167, 1136, 1101, 1062, 1020, 974, 925, 875, 821, 765, 708, 648, 591, 533, 475, 420, 367, 317, 270, 227, 188, 153, 123, 97, 75, 57, 42, 30, 21.7, 14.96, 10.03,6.55])/1320
    P_vvoa1 = np.asarray([1472, 1468, 1462, 1453, 1441, 1426, 1406, 1382, 1353, 1320, 1281, 1237, 1188, 1134, 1076, 1014, 947, 878, 807, 735, 662, 590, 520, 452, 388, 329, 275, 225.5, 182, 144.3, 112.3, 85.8, 63, 46.90, 33.45, 23.0, 15.75, 10.38, 6.71, 4.230, 2.558, 1.558])/1472

    #plt.figure(1)
    #plt.plot(V_VOA_3,P_3)
    #plt.show()
    VOA2_attenuation = 10*np.asarray(np.log10(P_vvoa2))
    VOA1_attenuation = 10*np.asarray(np.log10(P_vvoa1))
    attenuation_function_voa2 = interp1d(VOA2_attenuation, V_VOA_3, kind = 'cubic')
    attenuation_function_voa1 = interp1d(VOA1_attenuation, V_VOA_3, kind = 'cubic')
    c = -23# 
    a = -10
    b = c - a
    print(attenuation_function_voa1(a))
    print(attenuation_function_voa2(b))
    v_voa1 = attenuation_function_voa1(a)
    v_voa2 = attenuation_function_voa2(b)
    #Work on this tomorrow. Make function of linear powers leaving v_voa1, convert to attenuation vector for voa 1, do b = c - a, throw in loop (may need some extrapolations)
    #for i in list(range(0,10)):, 
    
    #prevent overloading by first attenuating larger part.
    if abs(v_voa1) >= abs(v_voa2):
        cDAQtest.cDAQ_write_DC(channel ="cDAQ1mod1/ao2", voltage = v_voa2)
        time.sleep(2)
        cDAQtest.cDAQ_write_DC(channel ="cDAQ1mod1/ao1", voltage = v_voa1)
    else:
        cDAQtest.cDAQ_write_DC(channel ="cDAQ1mod1/ao1", voltage = v_voa1)
        time.sleep(2)
        cDAQtest.cDAQ_write_DC(channel ="cDAQ1mod1/ao2", voltage = v_voa2)
    
    #    time.sleep(10)
    """
    writetask = nidaqmx.task.Task()
    writetask.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
    write_linspace = [-self.sweep_range*x/(2*self.sampels_per_channel) for x in range(1,self.sampels_per_channel+1)]
    writetask.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
    writetask.write(write_linspace, auto_start=True)
    writetask.wait_until_done()
    writetask.close()
                
    """
    """
    number_of_channels = 1
    number_of_samples = 20000
    sample_rate = 10000
    read_task = nidaqmx.Task()
    read_task.ai_channels.add_ai_voltage_chan("cDAQ1Mod3/ai0")
    
    values_read = np.zeros((number_of_channels, number_of_samples), dtype=np.float64)
    read_task.timing.cfg_samp_clk_timing(sample_rate , samps_per_chan = number_of_samples)
    reader = AnalogMultiChannelReader(read_task.in_stream)
    read_task.start()
    
    time.sleep(10)
    reader = AnalogMultiChannelReader(read_task.in_stream)
    reader.read_many_sample(values_read, number_of_samples_per_channel=number_of_samples,timeout=500)

    read_task.close()
    print(values_read)
    print("size of values_read", np.size(values_read))
    """