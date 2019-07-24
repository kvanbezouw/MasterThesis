import numpy as np
import sys
from scipy import signal 
import nidaqmx
from nidaqmx import constants
from nidaqmx.constants import FuncGenType
import time

import PyDAQmx as nidaq

#from PyQt5 import QtGui
#import time


class cDAQtest():
    def __init__(self):
        print("start everything")
        channelout = "cDAQ1Mod1/ao0"
        #x = self.cDAQ_read("cDAQ1Mod3/ai0", coupling = "DC")
        #print(x)
        #channel = task.ai_channels.add_ai_voltage_chan("cDAQ1Mod3/ai0") 
        #self.cDAQ_write_DC("cDAQ1Mod1/ao0", voltage = 5)
        #self.cDAQ_write_AC(channelout)
        #data = self.cDAQ_test()
    

    def cDAQ_read(self, channel):
        with nidaqmx.Task() as task:
            chan = task.ai_channels.add_ai_voltage_chan(channel)
            voltage = task.read()
        return voltage


    def cDAQ_read_cascade(self, channel, channel2 = None, coupling = "DC", Samples_Per_Sec = 1000):
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
        
    def cDAQ_write_AC(self, channel):
         with nidaqmx.Task() as task:
            task.ao_channels.add_ao_func_gen_chan(channel, amplitude=5.0,offset=0.0) #AC again doesnt seem to work (the entirefunction doesn't work. Not needed for thesis anyway)..

    def cDAQ_write_DC(self, channel, voltage):
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
    data = cDAQtest.cDAQ_read_cascade(channel = "cDAQ1Mod3/ai0", coupling = "DC", Samples_Per_Sec = 1000)
    print(len(data))
    #with nidaqmx.Task() as task:
    #    channel = task.ai_channels.add_ai_voltage_chan("cDAQ1Mod3/ai0")          #find channelnames by opening Ni MAX; Dev and Interfaces; Test Panel. Also possible to change dev name etc.
    #    print(task.read())

    #task.ai_channels.add_ai_voltage_chan("c-DAQ-91470/ai0")      #Dev name found in DeviceManager

    #task = nidaqmx.task.Task()
    
    #taskHandle = TaskHandle(0)
    #DAQmxCreateTask("",byref(taskHandle))
    #task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=-10.0, max_val=10.0)      #returns "ImportError: sys.meta_path is None, Python is likely shutting down" and some negative Status code (seems abnormal)
    
    
    #task.read()
    
    
    #cDAQtest.cDAQ_write_DC("cDAQ1Mod1/ao0", voltage = 5)
    
    #cDAQ=nidaqmx.task.Task() #create output task to supply to the cDAQ
    #b=cDAQ.ao_channels.add_ao_voltage_chan('Dev3/ao0,Dev3/ao1', min_val=-10.0, max_val=10.0)
    