from own_fine_scan_design import Ui_Form
from ControlLaserWorkingCode import TL6800
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
import nidaqmx
#from tools.trace_db_2_0 import Database
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSignal
import threading

import pyqtgraph as pg
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
import visa
import math
import operator
from scipy.interpolate import interp1d


sys.path.append("Z:\Lab\Pulsing_Setup\python")


class fine_scan_GUI(QtGui.QWidget):
    def __init__(self):

        #self.fs = FineScan()    Make this class later

        QtGui.QWidget.__init__(self, None)

        # Set up the user interface from Designer.
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.fs = FineScan()
        #Create the plot
        
        self.plot_wdg = pg.PlotWidget()
        self.plot_wdg.enableAutoRange()
        self.plot_wdg.showAxis('top', show=True)
        self.plot_wdg.showAxis('right', show=True)
        self.plot_wdg.showAxis('left', show=True)
        self.plot_wdg.showAxis('bottom', show=True)

        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
       
        self.ui.gridLayout.addWidget(self.plot_wdg, 0, 2, 8, 1)
        self.region = pg.LinearRegionItem()


        #connect all buttons to actions
        self.ui.btn_scan.clicked.connect(self.scan)
        #self.ui.btn_fit.clicked.connect(self.fit)
        self.ui.btn_save.clicked.connect(self.save_file)
        #self.ui.btn_save_txt.clicked.connect(self.save_txt)
        self.ui.comboBox_detection_fine_scan.activated[str].connect((lambda: self.switch_channel()))  
        self.ui.comboBox_plotting_fine_scan.activated[str].connect((lambda:self.plot_results()))


    def switch_channel(self):
        #Link this to the transmission data eventually
        text = self.ui.comboBox_detection_fine_scan.currentText()
        self.fs.channel = "cDAQ1Mod3/" + text
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading channel' + self.fs.channel)

    def switch_plotting(self):
        self.plot_results()

    def plot_results(self):

        self.plot_comboBox_text = self.ui.comboBox_plotting_fine_scan.currentText()

        try:
            self.plot_wdg.clear()
            #pg.setConfigOption('background', 0.3)
            #pg.setConfigOption('foreground', 'w')
       
            self.plot_wdg.addItem(self.region)
           
            self.plot_wdg.enableAutoRange()
            if(self.plot_comboBox_text == "Det vs V"):
                print("Plot detuning")
                self.plot_wdg.plot(self.fs.detuning_volt_vec, self.fs.detuning_vec, pen='w')
                self.plot_wdg.setLabel("bottom",text = "Applied voltage", units = "V")
                self.plot_wdg.setLabel("left", text = "Detuning", units = "MHz")
            if(self.plot_comboBox_text == "MZ vs V"):
                print("Plot MZ data")
                self.plot_wdg.plot(self.fs.xdata, self.fs.ydata, pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Step number", units = "V")
                self.plot_wdg.setLabel("left", text = "Voltage", units = "V")
            if(self.plot_comboBox_text == "Transmission vs Detuning"):
                print("Plot Transmission data")
                self.plot_wdg.plot(self.fs.interpolated_detuning(self.fs.sweep_linspace), pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)
  
        except:
            print("Plotting failed. Possibly you don't have data yet..")

    def show_data(self):
        self.ui.info_text.clear()
        self.ui.info_text.append("Wl    PD_disk \n")
        [self.ui.info_text.append("{}  {}".format(str(self.fs.xdata[i])[0:4], str(self.fs.ydata[i])[0:4])) for i in range(0,len(self.fs.xdata))]


    def save_file(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat)")
        try:
            f3= open(str(name[0]),'w')
            f3.writelines("Piezo (V)   Photodiode (from disk) output (V) \n")
            [f3.writelines("{} {}\n".format(str(self.fs.xdata[i]), str(self.fs.ydata[i]))) for i in range(0,len(self.fs.xdata))]
            f3.close()
            print("Succesfully wrote data to {}".format(name[0]))
        except FileNotFoundError:
            print("Chosen not to save data")
            pass


    def scan(self):
        # plot data
    
            #read the wavelength from the lineinput
        input_wavelength = float(self.ui.textedit_center_wavelength.text())
        input_wavelength = round(input_wavelength, 4) #prob not necessary
        print(input_wavelength)
        self.fs.fine_scan(input_wavelength)
        self.show_data()        
        self.plot_results()
   

class FineScan(TL6800):
    """FineScanning happens by changing the piezo voltage via the Frequency Modulation input channel. This, because the piezo voltage
    cannot be swept via laser commands. The FMinput channel is controlled by cDAQ ao channels."""
    
    def __init__(self):
        super().__init__()
        self.channelout_freq_mod = "cDAQ1Mod1/ao0"
        self.channel = "cDAQ1Mod3/ai3"
        self.list_with_ai_channels = [self.channel]
        self.number_of_ai_channels = len(self.list_with_ai_channels)
        self.TL6800_sweep_parameters(printbool = False)
        self.sweep_range = 2
        self.sampels_per_channel = 1000
        self.sweep_sampels_per_channel = 10000
        self.sweep_linspace = [x*self.sweep_range/self.sweep_sampels_per_channel - self.sweep_range/2  for x in range(1,self.sweep_sampels_per_channel+1)]

    def fine_scan(self, input_wavelength):
        print("input wavelenght")
        # check laser limits (with safety)
        if (input_wavelength > 1519.00) and (input_wavelength < 1571.00):
            #1 Hz. Well below the limit of 700 Hz for [-3 V -> +3 V]
            frequency = 0.5  
            #use this loop if you want to iterate more often over the same interval./
            for i in range(0,1):
                
                self.sampling_rate = frequency*self.sampels_per_channel
                self.sweep_sampling_rate = frequency*self.sweep_sampels_per_channel
                self.move_to_wavelength_start(input_wavelength)
                
                #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
                self.lower_freq_mod_voltage()
                time.sleep(0.5)
                print("start the fine sweep")
                self.sweep_freq_mod_voltage()
                time.sleep(0.5)
                print("fine sweep  fin'ed")
                self.get_freq_mod_voltage_back_to_zero()
                self.calculate_detuning(input_wavelength)

                #if(i == 0): self.fs_write_to_file(input_wavelength, i)

    def move_to_wavelength_start(self, input_wavelength):
        current_wavelength = float(self.TL6800_query_wavelength(printbool = False))
        time_needed = abs((current_wavelength-input_wavelength))+2
        self.TL6800_set_wavelength(input_wavelength, printbool = False)
        time.sleep(time_needed)

        #Laser is unstable during trackmode. Trackmode turns off some time after reaching the entered wavelength.  Have the finesweep
        #wait until trackmode is off (-> set wavelength == current wavelength) before entering the finesweep
        while True:
            trackmode = self.check_trackmode()
            if not trackmode: break

    
    def lower_freq_mod_voltage(self):
        writetask = nidaqmx.task.Task()
        writetask.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write_linspace = [-self.sweep_range*x/(2*self.sampels_per_channel) for x in range(1,self.sampels_per_channel+1)]
        writetask.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask.write(write_linspace, auto_start=True)
        writetask.wait_until_done()
        writetask.close()
               
    def sweep_freq_mod_voltage(self):
        #Move it to v_max
        sweeptask_ai,sweeptask_ao  = self.open_tasks(number_of_tasks = 2)
        self.add_ai_channels_to_task(sweeptask_ai)
        self.add_ao_channel_to_task(sweeptask_ao)
        values_read = np.zeros((self.number_of_ai_channels,self.sweep_sampels_per_channel), dtype=np.float64)
        reader = AnalogMultiChannelReader(sweeptask_ai.in_stream)
        sweeptask_ao.write(self.sweep_linspace, auto_start = False)
        self.run_piezo_sweep_and_wait_until_done(sweeptask_ai, sweeptask_ao)
        self.process_data(reader, values_read)
        self.close_tasks([sweeptask_ai,sweeptask_ao])

 
    def get_freq_mod_voltage_back_to_zero(self):
        #Move it back to 0V
        writetask_2 = nidaqmx.task.Task()
        writetask_2.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write2_linspace = [-x*self.sweep_range/(2*self.sampels_per_channel)+self.sweep_range/2 for x in range(1,self.sampels_per_channel+1)]
        writetask_2.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask_2.write(write2_linspace, auto_start = False)

        writetask_2.start()
        writetask_2.wait_until_done()
        writetask_2.close()
        
    def add_ai_channels_to_task(self, task_ai):
        for channels in self.list_with_ai_channels:
            task_ai.ai_channels.add_ai_voltage_chan(str(channels), min_val=-10, max_val=10)
        task_ai.timing.cfg_samp_clk_timing(self.sweep_sampling_rate , source='ao/SampleClock', samps_per_chan = self.sweep_sampels_per_channel)
        

    def add_ao_channel_to_task(self, task_ao):
        task_ao.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        task_ao.timing.cfg_samp_clk_timing(self.sweep_sampling_rate , samps_per_chan = self.sweep_sampels_per_channel)

    def process_data(self, reader, values_read):
        expected_max_time = self.sweep_sampels_per_channel/self.sweep_sampling_rate
        reader.read_many_sample(values_read, number_of_samples_per_channel=self.sweep_sampels_per_channel,timeout=expected_max_time+1)
        #read_data = sweeptask_ai.read(number_of_samples_per_channel=self.sweep_sampels_per_channel)
        self.ydata = np.asarray(values_read[0])
        self.xdata = np.asarray(self.sweep_linspace)

    @staticmethod
    def run_piezo_sweep_and_wait_until_done(task_ai, task_ao):
        task_ai.start()
        task_ao.start()
        task_ai.wait_until_done()
        task_ao.wait_until_done()

    @staticmethod
    def open_tasks(number_of_tasks):
        tasks = []
        for i in range(0, number_of_tasks):
            tasks.append(nidaqmx.task.Task())
        if number_of_tasks == 1:
            return tasks[0]
        if number_of_tasks == 2:
            return tasks[0], tasks[1]

    @staticmethod
    def close_tasks(tasks):
        for task in tasks:
            task.close()


    def check_trackmode(self):
        time.sleep(0.2)
        trackmode = self.TL6800_query_trackmode(printbool = False)
        print("trackmode tested")
        return trackmode
            
    def fs_write_to_file(self, input_wavelength, i):
        try:
            # Create target Directory
            os.mkdir(r'C:\Users\LocalAdmin\Documents\KeesFiles\MachZenderTesting\lambda' + str(input_wavelength))
            print("Directory created ") 
        except FileExistsError:
            print("Directory already exists")

        try:
            os.mkdir(r'C:\Users\LocalAdmin\Documents\KeesFiles\MachZenderTesting\lambda' + str(input_wavelength) + r'\volt_range' + str(self.sweep_range))
            print("Directory volt created ") 
        except FileExistsError:
                    print("Directory volt already exists")
            
        with open(r'C:\Users\LocalAdmin\Documents\KeesFiles\MachZenderTesting\lambda' + str(input_wavelength) + r'\volt_range' + str(self.sweep_range) +  r'\xdata' + str(i) + '.csv', 'w+') as f1:
            f1.writelines("%s\n" % points for points in self.xdata)
            f1.close()

        with open(r'C:\Users\LocalAdmin\Documents\KeesFiles\MachZenderTesting\lambda' + str(input_wavelength) + r'\volt_range' + str(self.sweep_range) +  r'\ydata' + str(i) + '.csv', 'w+') as f2:
            f2.writelines("%s\n" % points for points in self.ydata)
            f2.close()
                

    #r in front makes sure that backslashers are not interpreted as a special char

    def calculate_detuning(self, input_wavelength):

        V_range = self.sweep_range        #sweep_range
        L = self.sweep_sampels_per_channel            #Amount of steps
        T_v = V_range/L      #period (voltage per step). Was 2 mV
        F_s = 1/T_v           #Sampling frequency (step per voltage)
        volt_step_size = 0.010 #V
        step_size = self.from_volt_to_steps(volt_step_size, F_s)
        volt_bin_size = 0.2 #V
        bin_size = self.from_volt_to_steps(volt_bin_size, F_s) 
        n=1024
        max_it = int((V_range -bin_size* T_v)/(step_size*T_v))
        #print(self.sweep_linspace)

        x = range(L)
        y = range(L)
        freq_midpoint = []
        freq_max = []
        print("got to the pre-detuning loop?")
     
        #Calculate all the freq changes per volt, evaulated at certain points. FFT taken over a small surrounding interval
        for i in range(0,max_it): 
            start_point = step_size*i
        
            x_slice = self.xdata[start_point:(start_point+bin_size)]
            y_slice = self.ydata[start_point:(start_point+bin_size)]
         
            y_fft = np.fft.fft(y_slice,n)
          
            PSD2 = abs(y_fft/L)      #doublesided PSD
            PSD1 = PSD2[0:int(L/2)]
            PSD1[1:] = 2*PSD1[1:]
          
           
            f = [F_s*k/L for k in range(0,int(L/2))]
            #Tick the offset peak delta peak to zero. (all values below 10 V^-1, that is)
            PSD1[0:20] = 0
            
            index, value = max(enumerate(PSD1), key = operator.itemgetter(1))
            freq_midpoint.append(-V_range/2+T_v*(start_point+math.ceil(bin_size/2)))
            freq_max.append(f[index])

        print("got post the pre-detuning loop?")

        self.freq_max = np.asarray(freq_max)
        self.freq_midpoint = np.asarray(freq_midpoint)
        #Detuning is determined at the edges of the bins
        detuning_volt_vect= np.linspace(-V_range/2+T_v*(bin_size/2-step_size/2),V_range/2-T_v*(bin_size/2+step_size/2),max_it+1)

        detuning_summed = []
        detuning_summed.append(0)
        #Calculate detuning
        for i in range(0, max_it):
            detuning_summed.append(detuning_summed[i] + self.freq_max[i]*step_size*T_v*98.5)   #MHz. MZ calibrated to detune 98.5 MHz per cycle
        self.detuning_volt_vec = np.asarray(detuning_volt_vect)
        self.detuning_vec = np.asarray(detuning_summed)
        print(self.detuning_volt_vec[1:5])

        #Needed to properly allign detuning and PD data.
        """
        Story behind is, that transmission is only logged during the sweep, in the sweep_range. These voltages should be converted to detunings. 
        Matching the transmissions as y-data and the detunings as x-data should be able to plot the transmission vs detuning.
        """
        interpolated_detuning = interp1d(self.detuning_volt_vec, self.detuning_vec, kind = 'cubic') #valid from
        min_interpolation = np.amin(self.detuning_volt_vec)
        max_interpolation = np.amax(self.detuning_volt_vec)
        print("min interpol value is ", min_interpolation, " and max is ", max_interpolation)      
        inrange_indices = [index for index,volt in enumerate(self.sweep_linspace) if (volt < max_interpolation and volt > min_interpolation)]
        min_index = np.amin(inrange_indices)
        max_index = np.amax(inrange_indices)
        #print(inrange_indices)
        in_range_volt_vec = self.sweep_linspace[min_index:max_index]
        #get corresponding detunings
        #print(interpolated_detuning(in_range_volt_vec))
        #Now this should be able to be linked to the transmissions
        
        #print("detuning volt vec\n", self.detuning_volt_vec)
        #print("Hoping 1\n", interpolated_detuning(self.detuning_volt_vec))
        #print("sweep_linspace\n", self.sweep_linspace)
        #print("Hoping 2 \n", interpolated_detuning(self.sweep_linspace))

    @staticmethod
    def from_volt_to_steps(to_convert, sampling_rate):
        step_size = to_convert * sampling_rate
        if not step_size.is_integer():
            print("be careful, conversion from volts to steps is not an integer, which may or may not slightly distort all calibrations")
        steps = int(step_size)
        return steps

        


def main():
    app = QtGui.QApplication(sys.argv)
    myWidget = QtGui.QMainWindow()
    # main layout
    mainLayout = QtGui.QVBoxLayout()
    # add all main to the main vLayout
    mainLayout.addWidget(fine_scan_GUI())
    # self.mainLayout.addWidget(self.scrollArea)
    # central widget
    centralWidget = QtGui.QWidget()
    centralWidget.setLayout(mainLayout)
    # set central widget
    myWidget.setCentralWidget(centralWidget)


    myWidget.show()
    app.exec_()


if __name__ == '__main__':
    main()
