import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
from LASERCONTROL.Tl6800control import TL6800
from yellow_fine_scan_design import Ui_Form
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
import nidaqmx
#from tools.trace_db_2_0 import Database
from PyQt5 import QtGui, QtCore
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
from math import ceil, floor
from DRIVERS.data_fitting import Fitting_Tools
from DRIVERS.power_sweep import Power_Sweep
import csv
from itertools import zip_longest
from timeit import default_timer as timer
from DRIVERS.yellow_cDAQ import cDAQtest

class fine_scan_GUI(QtGui.QWidget):
    def __init__(self):

        #self.fs = FineScan()    Make this class later

        QtGui.QWidget.__init__(self, None)

        # Set up the user interface from Designer.
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.fs = FineScan()
        self.fit = Fitting_Tools()
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
        self.ui.btn_fine_scan_loop.clicked.connect(self.scan_loop)
        self.ui.btn_power_measurement.clicked.connect(self.power_measurement)
        #self.ui.btn_fit.clicked.connect(self.fit)
        self.ui.btn_save.clicked.connect(self.save_file)
        self.ui.btn_save_fit.clicked.connect(self.save_file_fitted)
        self.ui.btn_fit.clicked.connect(self.fit_model)
        self.ui.btn_save_power_meas.clicked.connect(self.save_file_power_meas)
        #self.ui.btn_save_txt.clicked.connect(self.save_txt)
        self.ui.comboBox_pd_detection_fine_scan.activated[str].connect((lambda: self.switch_channel_pd())) 
        self.ui.comboBox_MZ_detection.activated[str].connect((lambda: self.switch_channel_mz()))
        self.ui.comboBox_test_detection.activated[str].connect((lambda: self.switch_channel_test()))
        self.ui.comboBox_plotting_fine_scan.activated[str].connect((lambda: self.plot_results()))

    def switch_channel_pd(self):
        #Link this to the transmission data eventually
        text = self.ui.comboBox_pd_detection_fine_scan.currentText()
        self.fs.channel_pd = "cDAQ1Mod3/" + text
        self.fs.update_channels()
        print('now reading channel' + self.fs.channel_pd)

    def switch_channel_mz(self):
        #Link this to the transmission data eventually
        text = self.ui.comboBox_MZ_detection.currentText()
        self.fs.channel_mz = "cDAQ1Mod3/" + text
        self.fs.update_channels()
        print('now reading channel' + self.fs.channel_mz)

    def switch_channel_test(self):
        #Link this to the transmission data eventually
        text = self.ui.comboBox_test_detection.currentText()
        self.fs.channel_test = "cDAQ1Mod3/" + text
        self.fs.update_channels()

        print('now reading channel' + self.fs.channel_test)

    def switch_plotting(self):
        self.plot_results()

    def plot_results(self, clear = True):
        self.plot_comboBox_text = self.ui.comboBox_plotting_fine_scan.currentText()
        if clear:
            self.plot_wdg.clear()
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
        self.plot_wdg.enableAutoRange()
        try: 
            if(self.plot_comboBox_text == "Det vs V"):

                print("Plot detuning")
                self.plot_wdg.plot(self.fs.detuning_volt_vec, self.fs.detuning_vec, pen='w')
                self.plot_wdg.setLabel("bottom",text = "Applied voltage", units = "V")
                self.plot_wdg.setLabel("left", text = "Detuning", units = "MHz")

            if(self.plot_comboBox_text == "MZ vs V"):
                print("Plot MZ data")

                self.show_data(self.fs.sweep_linspace, self.fs.ydata[1])
                self.plot_wdg.plot(self.fs.xdata, self.fs.ydata[1], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Applied voltage", units = "V")
                self.plot_wdg.setLabel("left", text = "PD output", units = "V")

            if(self.plot_comboBox_text == "Tapered vs detuning"):
                print("Plot Transmission data")
                #-1 * self.fs.sweep_linspace. REMOVE LATER !!
                self.plot_wdg.plot(self.fs.interpolated_detuning_function(self.fs.sweep_linspace[self.fs.detuning_min_index:self.fs.detuning_max_index]),self.fs.ydata[0][self.fs.detuning_min_index:self.fs.detuning_max_index], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)

            if(self.plot_comboBox_text == "PDdisk vs V"):
                try:
                    self.show_data(self.fs.sweep_linspace, self.fs.ydata[0])
                except AttributeError:
                    print("you have no data to show yet, or your data is just bad main")
                print("Plot Transmission data")
                
                self.plot_wdg.setLabel("bottom",text = "Applied Voltage", units = "V")
                self.plot_wdg.setLabel("left", text = "PD Voltage ", units = "V")
                self.plot_wdg.plot(self.fs.sweep_linspace, self.fs.ydata[0], pen = 'w')

            if(self.plot_comboBox_text == "Test vs V"):
                print("Plot Transmission data")
                self.show_data(self.fs.sweep_linspace, self.fs.ydata[2])
                self.plot_wdg.plot(self.fs.sweep_linspace, self.fs.ydata[2], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Applied Voltage", units = "V")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)

            if(self.plot_comboBox_text == "power_scan_first"):
                self.plot_wdg.plot(self.fs.ps_xdata_detuning[0], self.fs.ps_ydata_detuning[0], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)
        
            if(self.plot_comboBox_text == "power_scan_middle"):
                middle_point = round(len(self.fs.ps_xdata_detuning)/2)
                self.plot_wdg.plot(self.fs.ps_xdata_detuning[middle_point], self.fs.ps_ydata_detuning[middle_point], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)

            if(self.plot_comboBox_text == "power_scan_last"):
                self.plot_wdg.plot(self.fs.ps_xdata_detuning[-3], self.fs.ps_ydata_detuning[-3], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)

            if(self.plot_comboBox_text == "power_scan_first_run_2"):
                self.plot_wdg.plot(self.fs.ps_xdata_detuning[-2], self.fs.ps_ydata_detuning[-2], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)
            
            if(self.plot_comboBox_text == "power_scan_ref"):
                self.plot_wdg.plot(self.fs.ps_xdata_detuning[-1], self.fs.ps_ydata_detuning[-1], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Detuning", units = "MHz")
                self.plot_wdg.setLabel("left", text = "Transmission", units = None)
        except:
            print("Plotting failed. Possibly you don't have data yet..")

    def show_data(self, xdata, ydata):
        self.ui.info_text.clear()
        self.ui.info_text.append("Wl    V \n")
        [self.ui.info_text.append("{}  {}".format(str(xdata[i])[0:4], str(ydata[i])[0:4])) for i in range(0,len(self.fs.xdata))]


    def save_file_fitted(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat *.csv)")
        try:
            data_compiled = [list(self.fs.xdata), list(self.fs.ydata[0]), list(self.fs.ydata[1]), list(self.x_data_detuning), list(self.y_data_detuning), list(self.final_fit), list(self.fs.params)]
            f1 = open(str(name[0]),'w')
            wr = csv.writer(f1)
            f1.writelines("xdata (V), PD_tapfib (V), MZ (V), xdata (MHz), ydata (MHz), fit (V vs MHz), fit parameters (amplitude (V) center (MHZ), linewidth (MHz) offset (V) linear noise coeff (V/MHZ)) \n")
            
            for values in zip_longest(*data_compiled):
                wr.writerow(values)
            f1.close()
            print("Succesfully wrote data to {}".format(name[0]))
        except FileNotFoundError:
            print("Chosen not to save data")
            pass
        except AttributeError:
            print("Caught Attribute error. Maybe you don't have data yet or haven't fitted yet.")

    def save_file(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat *.csv)")
        try:
            data_compiled = [list(self.fs.xdata), list(self.fs.ydata[0]), list(self.fs.ydata[1]), list(self.fs.x_data_detuning), list(self.fs.y_data_detuning)]
            f3= open(str(name[0]),'w')
            wr = csv.writer(f3)
            f3.writelines("xdata (V) , PD_tapfib (V), MZ (V), xdata (MHz), ydata (MHz),  \n")
            for values in zip_longest(*data_compiled):
                wr.writerow(values)
            f3.close()
            print("Succesfully wrote data to {}".format(name[0]))
        except FileNotFoundError:
            print("Chosen not to save data")
            pass
        except AttributeError:
            print("Caught Attribute error. Maybe you don't have data yet.")

    def save_file_power_meas(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat *.csv)")
               
        try:
            self.amplification = round(float(self.ui.lineEdit_amplification.text())) #This is work in progress
            for i in list(range(0,len(self.fs.ps_ydata)))   : #plus 1 because of reference thingy
                data_compiled = [list(self.fs.ps_xdata[i]), list(self.fs.ps_ydata[i][0]), list(self.fs.ps_ydata[i][1]), list(self.fs.ps_xdata_detuning[i]), list(self.fs.ps_ydata_detuning[i]), list(self.fs.params_power_sweep[i]), list(self.fs.power_linspace), list([self.amplification])]
                f3= open(str(name[0])+ str(i),'w')
                wr = csv.writer(f3)
                f3.writelines("xdata (V) , PD_tapfib (V), MZ (V), xdata (MHz), ydata (MHz), fit parameters (amplitude (V) center (MHZ) linewidth (MHz) offset (V) linear noise coeff (V/MHZ)), power linspace (mW), amplification \n")
                for values in zip_longest(*data_compiled):
                    wr.writerow(values)
                f3.close()
            print("Succesfully wrote data to {}".format(name[0]))
        except FileNotFoundError:
            print("Chosen not to save data")
            pass
        except AttributeError:
            print("Caught Attribute error. Maybe you don't have power meas data yet.")

    def fit_model(self):
            self.x_data_detuning = self.fs.interpolated_detuning_function(self.fs.sweep_linspace[self.fs.detuning_min_index:self.fs.detuning_max_index])
            self.y_data_detuning = self.fs.ydata[0][self.fs.detuning_min_index:self.fs.detuning_max_index]
            x_data = np.asarray(self.x_data_detuning)
            y_data = np.asarray(self.y_data_detuning)
            self.fs.params = self.fit.fit_data(x_data, y_data)
            print("fit parameters (amplitude, center, linewidth, offset, linear noise) are: ( {} ) respectively".format(str(self.fs.params[0:5])))
            pars_lorentz = self.fs.params[0:3]
            pars_background = self.fs.params[3:]
            lor_peak_1 = self.fit._1Lorentz(x_data, *pars_lorentz)      
            self.final_fit = np.asarray(self.fit.single_dip_fit_function_lin(x_data, *self.fs.params))
            background_noise = np.asarray(self.fit.linear(x_data, *pars_background))
            self.plot_wdg.plot(x_data, self.final_fit, pen = "r")
            self.plot_wdg.plot(x_data, background_noise, pen = "y")

    def power_measurement(self):
        
        try:
            
            input_wavelength = float(self.ui.textedit_center_wavelength.text())
            self.fs.process_sweep_range(float(self.ui.lineEdit_sweep_range.text()))
            self.fs.ps_scaling = float(self.ui.ps_scaling.text())
            assert input_wavelength >= 1520 and input_wavelength <= 1570
            assert self.fs.sweep_range <= 6
            assert len(self.fs.list_with_ai_channels) == len(set(self.fs.list_with_ai_channels))
            assert self.fs.ps_scaling <= 1
            print("Scan around {} nm". format(input_wavelength)) 
            self.fs.process_expansion(int(self.ui.lineEdit_avg_over_x_steps.text()))                #just to be sure
            print("Sweep range {} V".format(self.fs.sweep_range))
            self.fs.ps_data_points = 20
            self.fs.power_scan(input_wavelength, voa1_channel = "cDAQ1Mod1/ao1", voa2_channel = "cDAQ1Mod1/ao2", data_points = self.fs.ps_data_points, write_to_file_bool = "off")
            print("Power measurement finished")
            #self.plot_results()
        except AssertionError:
            print("One of your Fine Scan line edit values is out of range/erronous, or you've defined multiple ai tasks on the same channel/")
            pass
        except ValueError:
            print("U didnt enter a proper value for the wavelength, did you?")
            pass

    def scan_loop(self):
        input_wavelength = float(self.ui.textedit_center_wavelength.text())
        try:
            
            self.fs.process_sweep_range(float(self.ui.lineEdit_sweep_range.text()))
            assert input_wavelength >= 1520 and input_wavelength <= 1570
            assert self.fs.sweep_range <= 6
            assert len(self.fs.list_with_ai_channels) == len(set(self.fs.list_with_ai_channels))

            print("Scan around {} nm". format(input_wavelength)) 
            self.fs.process_expansion(int(self.ui.lineEdit_avg_over_x_steps.text()))                #just to be sure
            print("Sweep range {} V".format(self.fs.sweep_range))
            self.fs.fine_scan_loop(input_wavelength)    
            self.plot_results()
        except AssertionError:
            print("One of your Fine Scan line edit values is out of range/erronous, or you've defined multiple ai tasks on the same channel/")
            pass
        except ValueError:
            print("U didnt enter a proper value for the wavelength, did you?")
            pass

    def scan(self):
        
        try:
            input_wavelength = float(self.ui.textedit_center_wavelength.text())
            self.fs.process_sweep_range(float(self.ui.lineEdit_sweep_range.text()))
            assert input_wavelength >= 1520 and input_wavelength <= 1570
            assert self.fs.sweep_range <= 6
            assert len(self.fs.list_with_ai_channels) == len(set(self.fs.list_with_ai_channels))

            print("Scan around {} nm". format(input_wavelength)) 
            self.fs.process_expansion(int(self.ui.lineEdit_avg_over_x_steps.text()))                #just to be sure
            print("Sweep range {} V".format(self.fs.sweep_range))
            self.fs.fine_scan(input_wavelength)    
            self.plot_results()
        except AssertionError:
            print("One of your Fine Scan line edit values is out of range/erronous, or you've defined multiple ai tasks on the same channel/")
            pass
        except ValueError:
            print("U didnt enter a proper value for the wavelength, did you?")
            pass
    
    def scan_reversed(self):
        input_wavelength = float(self.ui.textedit_center_wavelength.text())
        try:
            
            self.fs.process_sweep_range(float(self.ui.lineEdit_sweep_range.text()))
            assert input_wavelength >= 1520 and input_wavelength <= 1570
            assert self.fs.sweep_range <= 6
            assert len(self.fs.list_with_ai_channels) == len(set(self.fs.list_with_ai_channels))

            print("Scan around {} nm". format(input_wavelength)) 
            self.fs.process_expansion(int(self.ui.lineEdit_avg_over_x_steps.text()))
            print("Sweep range {} V".format(self.fs.sweep_range))
            self.fs.fine_scan_reversed(input_wavelength)    
            self.plot_results()
        except AssertionError:
            print("One of your Fine Scan line edit values is out of range/erronous, or you've defined multiple ai tasks on the same channel/")
            pass
   
#inheret all class variables and functions from the laser and the power sweep
class FineScan(TL6800, Power_Sweep):
    """FineScanning happens by changing the piezo voltage via the Frequency Modulation input channel. This, because the piezo voltage
    cannot be swept via laser commands. The FMinput channel is controlled by cDAQ ao channels."""
    
    def __init__(self):
        super().__init__()
        self.channelout_freq_mod = "cDAQ1Mod1/ao0"
        self.channel_pd = "cDAQ1Mod3/ai0"
        self.channel_mz = "cDAQ1Mod3/ai0"
        self.channel_test = "cDAQ1mod3/ai0"
        self.channel_switch = "cDAQ1mod1/ao3"
        self.list_with_ai_channels = [self.channel_pd, self.channel_mz, self.channel_test]
        self.number_of_ai_channels = len(self.list_with_ai_channels)
        
        self.TL6800_sweep_parameters(printbool = False)
        self.sweep_range = 2
        self.sampels_per_channel = 1000
        self.sweep_sampels_per_channel = 9000
        self.ps_scaling = 0.1
        self.sweep_linspace = [x*self.sweep_range/self.sweep_sampels_per_channel - self.sweep_range/2  for x in range(1,self.sweep_sampels_per_channel+1)] 
        self.ydata = np.zeros([self.number_of_ai_channels,self.sweep_sampels_per_channel])
        
        self.average_over_n_points = 4
  
        self.process_expansion(self.average_over_n_points)

    def update_channels(self):
        self.list_with_ai_channels = [self.channel_pd, self.channel_mz, self.channel_test]
        
    def process_expansion(self, avg_over_n_points):
        #Talking multiple samples per voltage requires some matrices to be expanded
        if not self.average_over_n_points:                                 #don't print when starting up GUI
            print("Avg over {} points now". format(self.average_over_n_points))
            self.average_over_n_points = int(avg_over_n_points)
        else: 
            self.average_over_n_points = int(avg_over_n_points)
        self.sweep_sampels_per_channel_expanded = self.average_over_n_points * self.sweep_sampels_per_channel
        self.sweep_linspace_expanded = self.expand_matrix(self.sweep_linspace, rep_n_times= self.average_over_n_points)
        self.ydata_expanded = np.zeros([self.number_of_ai_channels, self.sweep_sampels_per_channel_expanded])

    def fine_scan(self, input_wavelength):
        #Have the laser perform a fine scan
        #1 Hz. Well below the limit of 700 Hz for [-3 V -> +3 V] (see use manual of the laser)
        frequency = 0.5  
        #use this loop if you want to iterate more often over the same interval./
        self.sampling_rate = frequency*self.sampels_per_channel
        self.sweep_sampling_rate = frequency*self.sweep_sampels_per_channel
        self.sweep_sampling_rate_expanded = frequency * self.sweep_sampels_per_channel_expanded
        self.move_to_wavelength_start(input_wavelength) 
    
        for i in range(0,1):
            try:
                assert self.sweep_sampling_rate_expanded < 165000        #Maximum sampling rate
            
                #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
                self.lower_freq_mod_voltage()
                time.sleep(0.1)
                self.xdata, self.ydata =  self.sweep_freq_mod_voltage()
                time.sleep(0.1)
                self.get_freq_mod_voltage_back_to_zero()
                self.calculate_detuning(input_wavelength, self.xdata, self.ydata)
                #if(i == 0): self.fs_write_to_file(input_wavelength, i)


                self.x_data_detuning = self.interpolated_detuning_function(self.sweep_linspace[self.detuning_min_index:self.detuning_max_index])
                self.y_data_detuning = self.ydata[0][self.detuning_min_index:self.detuning_max_index]
                x_data = np.asarray(self.x_data_detuning)
                y_data = np.asarray(self.y_data_detuning)
                self.params = Fitting_Tools.fit_data(x_data, y_data)
               
                print("fit parameters (amplitude, center, linewidth, offset, linear noise) are: ( {} ) respectively".format(str(self.params[0:5])))
      
                
            except AssertionError:                   #This prevents the laser from starting the fien scan
                raise AssertionError

    def fine_scan_reversed(self, input_wavelength):
            #Have the laser perform a fine scan
            #1 Hz. Well below the limit of 700 Hz for [-3 V -> +3 V] (see use manual of the laser)
            frequency = 0.5  
            #use this loop if you want to iterate more often over the same interval./
            self.sampling_rate = frequency * self.sampels_per_channel
            self.sweep_sampling_rate = frequency * self.sweep_sampels_per_channel
            self.sweep_sampling_rate_expanded = frequency * self.sweep_sampels_per_channel_expanded
            self.move_to_wavelength_start(input_wavelength) 
         
            for i in range(0,1):
                try:
                    assert self.sweep_sampling_rate_expanded < 165000        #Maximum sampling rate
                
                    #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
                    self.lower_freq_mod_voltage_reversed()
                    time.sleep(0.1)
                    self.sweep_freq_mod_voltage_reversed()
                    time.sleep(0.1)
                    self.get_freq_mod_voltage_back_to_zero_reversed()
                    self.calculate_detuning(input_wavelength)
                    #if(i == 0): self.fs_write_to_file(input_wavelength, i)


                    self.x_data_detuning = self.interpolated_detuning_function(self.sweep_linspace[self.detuning_min_index:self.detuning_max_index])
                    self.y_data_detuning = self.ydata[0][self.detuning_min_index:self.detuning_max_index]
                    x_data = np.asarray(self.x_data_detuning)
                    y_data = np.asarray(self.y_data_detuning)
                    self.params = Fitting_Tools.fit_data(x_data, y_data)
                
                    print("fit parameters (amplitude, center, linewidth, offset, linear noise) are: ( {} ) respectively".format(str(self.params[0:5])))
        
                    
                except AssertionError:                   #This prevents the laser from starting the fien scan
                    raise AssertionError
    
    def fine_scan_loop(self, input_wavelength):
            #Have the laser perform a fine scan
            #1 Hz. Well below the limit of 700 Hz for [-3 V -> +3 V] (see use manual of the laser)
            frequency = 0.5 
            #use this loop if you want to iterate more often over the same interval./
            self.sampling_rate = frequency*self.sampels_per_channel
            self.sweep_sampling_rate = frequency*self.sweep_sampels_per_channel
            self.sweep_sampling_rate_expanded = frequency * self.sweep_sampels_per_channel_expanded
            self.params = []
            times=  [0]
            self.move_to_wavelength_start(input_wavelength) 
            for i in range(0,80):
                try:
                    assert self.sweep_sampling_rate_expanded < 165000        #Maximum sampling rate
                    
                    start = timer()
                    #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
                    self.lower_freq_mod_voltage()
                    time.sleep(0.1)
                    x_data, y_data =  self.sweep_freq_mod_voltage()
                    time.sleep(0.1)
                    self.get_freq_mod_voltage_back_to_zero()
                    interpolated_detuning_function = self.calculate_detuning(input_wavelength, x_data, y_data)
                    #if(i == 0): self.fs_write_to_file(input_wavelength, i)


                    x_data_detuning = interpolated_detuning_function(self.sweep_linspace[self.detuning_min_index:self.detuning_max_index])
                    y_data_detuning = self.ydata[0][self.detuning_min_index:self.detuning_max_index]
                    x_data = np.asarray(x_data_detuning)
                    y_data = np.asarray(y_data_detuning)
                    self.params.append(list(Fitting_Tools.fit_data(x_data, y_data)))
                
                    print("fit parameters (amplitude, center, linewidth, offset, linear noise) are: ( {} ) respectively".format(str(self.params[i][0:5])))

                    time.sleep(10)
                    end = timer()   
                    print("time elapsed: {}".format(end-start))
                    times.append(times[i]+end-start)
                    print(times)
                    print(self.params)
                except AssertionError:                   #This prevents the laser from starting the fien scan
                    raise AssertionError
            print("params")
            print(self.params)

    def initiate_power_scan_variables_and_matrices(self):
        frequency = 0.5 
        self.sampling_rate = frequency*self.sampels_per_channel
        self.sweep_sampling_rate = frequency*self.sweep_sampels_per_channel
        self.sweep_sampling_rate_expanded = frequency * self.sweep_sampels_per_channel_expanded
        list_ps_x_data = []
        list_ps_y_data = []
        list_params_power_sweep =[]
        list_ps_x_data_detuning = []
        list_ps_y_data_detuning = []
        times=  [0]
        return list_ps_x_data, list_ps_y_data, list_params_power_sweep, list_ps_x_data_detuning, list_ps_y_data_detuning, times
        
    def power_scan(self, input_wavelength, voa1_channel, voa2_channel, data_points = 10, write_to_file_bool = "off"):
        #Have the laser perform a fine scan
        #1 Hz. Well below the limit of 700 Hz for [-3 V -> +3 V] (see use manual of the laser)
        
        #use this loop if you want to iterate more often over the same interval.
        list_ps_x_data, list_ps_y_data, list_params_power_sweep, list_ps_x_data_detuning, list_ps_y_data_detuning, times = self.initiate_power_scan_variables_and_matrices()
        self.power_linspace, self.attenuation_vector = self.create_non_linear_attenuation_vector(data_points = data_points, density_decrease = 4, scaling = self.ps_scaling, max_power = 3.3)
 
        
        max_attenuation = float(min(self.attenuation_vector))
        print("power scaling = {}".format(self.ps_scaling))
        print("power linspace: {}".format(str(3.3 * 10**(self.attenuation_vector/10))))
        
        self.move_to_wavelength_start(input_wavelength) #Moving  is innacurate; move to wavelength once, maybe correct for drifting afterwards
        try:
            assert self.sweep_sampling_rate_expanded < 165000     
            for i in range(0, len(self.attenuation_vector)):
                print("Run nr {}".format(i))
                #Maximum sampling rate
                v_voa1, v_voa2 = self.set_voas(max_attenuation = max_attenuation, attenuation_voa1 = self.attenuation_vector[i] , channel_voa1 = voa1_channel, channel_voa2 = voa2_channel)
                print("VOA1 = {}".format(v_voa1))
                print("VOA2 = {}".format(v_voa2))
                print("Power through tap. fib = {} mW".format(str(3.3 * 10**(self.attenuation_vector[i]/10)))) # 3.3 is calibrated by max power in current circuit.
                start = timer()
                if i == (len(self.attenuation_vector)-1):
                      #Give disk, piezo, voas some time to cool down, settle to new positions etc. etc.
                     print("i = {}, time to sleep so that voa_ 1,2 can adjust to their new values".format(i))
                     time.sleep(10)
                     print("slept")
                
               
                #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
                x_data, y_data = self.perform_single_run()
    
                interpolated_detuning_function = self.calculate_detuning(input_wavelength, x_data, y_data)
                #if(i == 0): self.fs_write_to_file(input_wavelength, i)
                list_ps_x_data.append(x_data)
                list_ps_y_data.append(y_data)
                list_ps_x_data_detuning.append(interpolated_detuning_function(self.sweep_linspace[self.detuning_min_index:self.detuning_max_index]))
                list_ps_y_data_detuning.append(y_data[0][self.detuning_min_index:self.detuning_max_index])
                x_data_fit = np.asarray(list_ps_x_data_detuning[i])
                y_data_fit = np.asarray(list_ps_y_data_detuning[i])
                list_params_power_sweep.append(Fitting_Tools.fit_data(x_data_fit, y_data_fit)) #now it fits a lorentzian. Maybe fit to toothsaw later on

                print("power sweep fit parameters (amplitude, center, linewidth, offset, linear noise) are: ( {} ) respectively".format(str(list_params_power_sweep[i][0:5])))

            
                end = timer()
                print("time elapsed: {}".format(end-start))
                times.append(times[i]+end-start)
                print("Elapsed time from start: {}".format(times[i]))
             #add another run for reference value. Add 3 dB to voa 2 to be sure that amp never overloads (=~ max loss through tap fib) 


            #One more test at max power at voa1 to see if anything changed
            print("Reference data is acquired after this message appears")
            v_voa1, v_voa2 = self.set_voas(max_attenuation = (max_attenuation - 3), attenuation_voa1 = self.attenuation_vector[5] , channel_voa1 = voa1_channel, channel_voa2 = voa2_channel)
            x_data, y_data = self.take_reference_scan()

            interpolated_detuning_function = self.calculate_detuning(input_wavelength, x_data, y_data)
            #Set Voas back to originaldo values so that voltages outside of power measurements are always consistent 
            v_voa1, v_voa2 = self.set_voas(max_attenuation = max_attenuation, attenuation_voa1 = self.attenuation_vector[1] , channel_voa1 = voa1_channel, channel_voa2 = voa2_channel)
            list_ps_x_data.append(x_data)
            list_ps_y_data.append(y_data)
            list_ps_x_data_detuning.append(interpolated_detuning_function(self.sweep_linspace[self.detuning_min_index:self.detuning_max_index]))
            list_ps_y_data_detuning.append(y_data[0][self.detuning_min_index:self.detuning_max_index])
            x_data_fit = np.asarray(list_ps_x_data_detuning[-1])
            y_data_fit = np.asarray(list_ps_y_data_detuning[-1])
            list_params_power_sweep.append(Fitting_Tools.fit_data(x_data_fit, y_data_fit)) #now it fits a lorentzian. Maybe fit to toothsaw later on

            self.params_power_sweep = np.asarray(list_params_power_sweep)
            self.ps_xdata_detuning = np.asarray(list_ps_x_data_detuning)
            self.ps_ydata_detuning = np.asarray(list_ps_y_data_detuning)
            self.ps_xdata = np.asarray(list_ps_x_data)
            self.ps_ydata = np.asarray(list_ps_y_data)

        except AssertionError:                   #This prevents the laser from starting the power sweep
            raise AssertionError
  
    def perform_single_run(self):
        time.sleep(0.3)
        self.lower_freq_mod_voltage() 
        time.sleep(0.1)
        x_data, y_data  = self.sweep_freq_mod_voltage()
        time.sleep(0.1)
        self.get_freq_mod_voltage_back_to_zero()
        return x_data, y_data

    def take_reference_scan(self):
        time.sleep(0.1)
        self.switch_switch(state = "Ref", channel = self.channel_switch)
        time.sleep(0.1)
        
        #Perform the piezo sweep (first from 0 to v_min, then from v_min to v_max, then from v_max to 0)
        self.lower_freq_mod_voltage()
        time.sleep(0.1)
        x_data, y_data  = self.sweep_freq_mod_voltage()
        
        time.sleep(0.1)
        self.get_freq_mod_voltage_back_to_zero()
        self.switch_switch(state = "Tap_fib", channel = self.channel_switch)
        return x_data, y_data


    @staticmethod
    def switch_switch(state, channel): #5V = through tap fib, 0V = ref
        try:
            state_str = str(state)
            assert state_str == "Tap_fib" or state_str == "Ref"
            
            if state_str == "Tap_fib":
                cDAQtest.cDAQ_write_DC(channel = channel, voltage = 5)
                print("Switch toggled to tap fib")
            if state_str == "Ref":
                cDAQtest.cDAQ_write_DC(channel = channel, voltage = 0)
                print("Switch toggled to reference fib")
        except AssertionError:
            print("Switch is fed the wrong state")
            raise AssertionError


    def process_sweep_range(self, new_range):
        #Changing sweep_range causes some variables to need to be updated
        self.sweep_range = new_range
        self.sweep_linspace = [x*self.sweep_range/self.sweep_sampels_per_channel - self.sweep_range/2  for x in range(1,self.sweep_sampels_per_channel+1)]
        self.sweep_linspace_expanded = self.expand_matrix(self.sweep_linspace, rep_n_times= self.average_over_n_points)

    def move_to_wavelength_start(self, input_wavelength):
        #Have the laser move to the wavelength to scan around
        current_wavelength = float(self.TL6800_query_wavelength(printbool = False))
        time_needed = abs((current_wavelength-input_wavelength))+2
        self.TL6800_set_wavelength(input_wavelength, printbool = False)
        time.sleep(time_needed)
        trackcounter = 0 
        #Laser is unstable during trackmode. Trackmode turns off some time after reaching the entered wavelength.  Have the finesweep
        #wait until trackmode is off (-> set wavelength == current wavelength) before entering the finesweep
        while True:
            trackmode = self.check_trackmode()
            trackcounter += 1
            if trackcounter >= 10:
                new_wavelength = input_wavelength + 1
                self.TL6800_set_wavelength(new_wavelength, printbool = False)            #sometimes it gets stuck and trackmode is never togelled off. Moving it away and then aiming again fixes this
                time.sleep(abs(new_wavelength-input_wavelength)+1)
                print("Trackmode got stuck. Try moving to input wavelength again")
                self.TL6800_set_wavelength(input_wavelength, printbool = False)
                time.sleep(2)
                trackcounter = 0
            if not trackmode: break

    def lower_freq_mod_voltage(self):
        #Move piezo from 0V to V_min
        writetask = nidaqmx.task.Task()
        writetask.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write_linspace = [-self.sweep_range*x/(2*self.sampels_per_channel) for x in range(1,self.sampels_per_channel+1)]
        writetask.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask.write(write_linspace, auto_start=True)
        writetask.wait_until_done()
        writetask.close()
               
    def sweep_freq_mod_voltage(self):
        #Move piezo from V_min to V_max
        sweeptask_ai,sweeptask_ao  = self.open_tasks(number_of_tasks = 2)
        self.add_ai_channels_to_task(sweeptask_ai, self.sweep_sampling_rate_expanded, self.sweep_sampels_per_channel_expanded)
        self.add_ao_channel_to_task(sweeptask_ao, self.sweep_sampling_rate_expanded, self.sweep_sampels_per_channel_expanded)
        values_read = np.zeros((self.number_of_ai_channels,self.sweep_sampels_per_channel_expanded), dtype=np.float64)
        reader = AnalogMultiChannelReader(sweeptask_ai.in_stream)
        sweeptask_ao.write(self.sweep_linspace_expanded, auto_start = False)
        self.run_piezo_sweep_and_wait_until_done(sweeptask_ai, sweeptask_ao)
        xdata, ydata = self.process_data(reader, values_read)
        self.close_tasks([sweeptask_ai,sweeptask_ao])
        return xdata, ydata
 
    def get_freq_mod_voltage_back_to_zero(self):
        #Move piezo back to 0V
        writetask_2 = nidaqmx.task.Task()
        writetask_2.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write2_linspace = [-x*self.sweep_range/(2*self.sampels_per_channel)+self.sweep_range/2 for x in range(1,self.sampels_per_channel+1)]
        writetask_2.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask_2.write(write2_linspace, auto_start = False)

        writetask_2.start()
        writetask_2.wait_until_done()
        writetask_2.close()

    def lower_freq_mod_voltage_reversed(self):
        #Move piezo from 0V to V_min
        writetask = nidaqmx.task.Task()
        writetask.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write_linspace =  [-1*-self.sweep_range*x/(2*self.sampels_per_channel) for x in range(1,self.sampels_per_channel+1)]
        writetask.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask.write(write_linspace, auto_start=True)
        writetask.wait_until_done()
        writetask.close()
               
    def sweep_freq_mod_voltage_reversed(self):
        #Move piezo from V_min to V_max
        sweeptask_ai,sweeptask_ao  = self.open_tasks(number_of_tasks = 2)
        self.add_ai_channels_to_task(sweeptask_ai, self.sweep_sampling_rate_expanded, self.sweep_sampels_per_channel_expanded)
        self.add_ao_channel_to_task(sweeptask_ao, self.sweep_sampling_rate_expanded, self.sweep_sampels_per_channel_expanded)
        values_read = np.zeros((self.number_of_ai_channels,self.sweep_sampels_per_channel_expanded), dtype=np.float64)
        reader = AnalogMultiChannelReader(sweeptask_ai.in_stream)
        self.sweep_linspace_expanded = -1* self.sweep_linspace_expanded 
        sweeptask_ao.write(self.sweep_linspace_expanded, auto_start = False)
        self.run_piezo_sweep_and_wait_until_done(sweeptask_ai, sweeptask_ao)
        xdata, ydata = self.process_data(reader, values_read)
        self.close_tasks([sweeptask_ai,sweeptask_ao])
        return xdata, ydata
 
    def get_freq_mod_voltage_back_to_zero_reversed(self):
        #Move piezo back to 0V
        writetask_2 = nidaqmx.task.Task()
        writetask_2.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        write2_linspace = [-1*(-x*self.sweep_range/(2*self.sampels_per_channel)+self.sweep_range/2) for x in range(1,self.sampels_per_channel+1)]
        writetask_2.timing.cfg_samp_clk_timing(self.sampling_rate , samps_per_chan = self.sampels_per_channel)
        writetask_2.write(write2_linspace, auto_start = False)
        writetask_2.start()
        writetask_2.wait_until_done()
        writetask_2.close()
        
    def add_ai_channels_to_task(self, task_ai, sampling_rate, sampels_per_channel):
        for channels in self.list_with_ai_channels:
            task_ai.ai_channels.add_ai_voltage_chan(str(channels), min_val=-10, max_val=10)
        task_ai.timing.cfg_samp_clk_timing(sampling_rate, source='ao/SampleClock', samps_per_chan = sampels_per_channel)
        
    def add_ao_channel_to_task(self, task_ao, sampling_rate, sampels_per_channel):
        task_ao.ao_channels.add_ao_voltage_chan(self.channelout_freq_mod, min_val = -self.sweep_range/2 - 0.1, max_val = self.sweep_range/2 + 0.1)
        task_ao.timing.cfg_samp_clk_timing(sampling_rate , samps_per_chan = sampels_per_channel)

    def process_data(self, reader, values_read):
        #convert raw data to xdata and ydata arrays.
        expected_max_time = self.sweep_sampels_per_channel_expanded/self.sweep_sampling_rate_expanded
        reader.read_many_sample(values_read, number_of_samples_per_channel= self.sweep_sampels_per_channel_expanded, timeout = expected_max_time+1)
        ydata = np.zeros([self.number_of_ai_channels, self.sweep_sampels_per_channel])
        for i in range(self.number_of_ai_channels):
            self.ydata_expanded[i] = values_read[i][:]
            for j in range(self.sweep_sampels_per_channel):
                start_index = j * self.average_over_n_points
                end_index =  start_index + self.average_over_n_points
                ydata[i][j] = np.mean(self.ydata_expanded[i][start_index:end_index])
        xdata = np.asarray(self.sweep_linspace)
        return xdata, ydata

    def check_trackmode(self):
        #Can't send new tasks to laser while trackmode is still on
        time.sleep(0.1)
        trackmode = self.TL6800_query_trackmode(printbool = False)
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

    def calculate_detuning(self, input_wavelength, xdata, ydata):
        #Converts MZ data to detuning data. Also creates continious function so that detuning can be evaluated anywhere within a certain voltage range.
        V_range = self.sweep_range        #sweep_range
        L = self.sweep_sampels_per_channel            #Amount of steps
        T_v = V_range/L      #period (voltage per step). Was 2 mV
        F_s = 1/T_v           #Sampling frequency (step per voltage)
        volt_step_size = 0.010 #V
        MZ_calibr = 98.5  #FSR of MZ is 98.5 MHz
        step_size = self.from_volt_to_steps(volt_step_size, F_s)
        volt_bin_size = 0.2 #V
        bin_size = self.from_volt_to_steps(volt_bin_size, F_s) 
        max_it = int((V_range -bin_size* T_v)/(step_size*T_v))
        #Calculate all the freq changes per volt, evaulated at certain points. FFT taken over a small surrounding interval (bin) and evaluated at certain voltages
        self.freq_midpoint, self.freq_max  = self.get_local_FSRs(xdata, ydata, L, V_range, T_v, F_s, bin_size, step_size, max_it)
        self.detuning_volt_vec, self.detuning_vec = self.get_cumulative_detuning(self.freq_max, V_range, T_v, bin_size, step_size, max_it, MZ_calibr)
        detuning_volt_vec, detuning_vec = self.get_cumulative_detuning(self.freq_max, V_range, T_v, bin_size, step_size, max_it, MZ_calibr)
        interpolated_detuning_function = self.make_a_continious_function_of_detuning_vs_voltage( detuning_volt_vec, detuning_vec)
        return interpolated_detuning_function
        

    def make_a_continious_function_of_detuning_vs_voltage(self,  detuning_volt_vec, detuning_vec):
        #Make a function of detuning of detuning vs voltage, so that all sample points can be linked to a detuning. Work on this later.
        self.interpolated_detuning_function = interp1d(self.detuning_volt_vec, self.detuning_vec, kind = 'cubic') #valid from
        interpolated_detuning_function = interp1d(self.detuning_volt_vec, self.detuning_vec, kind = 'cubic') 
        min_interpolation = np.amin(self.detuning_volt_vec)
        max_interpolation = np.amax(self.detuning_volt_vec)
   
        inrange_indices = [index for index,volt in enumerate(self.sweep_linspace) if (volt < max_interpolation and volt > min_interpolation)]
        self.detuning_min_index = np.amin(inrange_indices)
        self.detuning_max_index = np.amax(inrange_indices)

        in_range_volt_vec = self.sweep_linspace[self.detuning_min_index:self.detuning_max_index]
        return interpolated_detuning_function

        #Now this should be able to be linked to the transmissions
        
        #print("detuning volt vec\n", self.detuning_volt_vec)
        #print("Hoping 1\n", interpolated_detuning(self.detuning_volt_vec))
        #print("sweep_linspace\n", self.sweep_linspace)
        #print("Hoping 2 \n", interpolated_detuning(self.sweep_linspace))

    def get_local_FSRs(self, xdata, ydata, L, V_range, T_v, F_s, bin_size, step_size, max_it):
        #Local FSRs are evaluated by fourier transorming picking the most dominant frequency in a small interval around a certain sample point 
        #(after exclusion of the offset (dirac delta-ish in frequency space)
        n=self.next_power_of_2(L)            #needed to take n-point  fourier transform
        f = [F_s*k/n for k in range(0,int(n/2))]   #discrete FTT data points (up to f_nyquist + some padding for all n-L>0)
        ydata_without_offset = ydata[1] - np.mean(ydata[1])         #remove offset -> smaller deltapeak in results so that this is never calc'ed as f_max
        freq_midpoint = []                  #evaluate detuning at these points
        freq_max = []                        #used to store local FSRs
        
        for i in range(0,max_it+2): 
            start_point = step_size*i
            x_slice = xdata[start_point:(start_point+bin_size)]
            y_slice = ydata_without_offset[start_point:(start_point+bin_size)] #ydata[1] is MZ data
            PSD_1 = self.get_one_sided_PSD(y_slice, n)
            PSD_1[0:20] = 0
            index, value = max(enumerate(PSD_1), key = operator.itemgetter(1))
            freq_midpoint.append(-V_range/2+T_v*(start_point+math.ceil(bin_size/2)))
            freq_max.append(f[index])
            
        np_freq_midpoint = np.asarray(freq_midpoint)
        np_freq_max = np.asarray(freq_max)

        
        return np_freq_midpoint, np_freq_max

    @staticmethod
    def get_cumulative_detuning(local_FSRs, V_range, T_v, bin_size, step_size, max_it, MZ_calibr):
        #Stepwisily integrate the local_FSRs to get the total detuning at a certain point. Set V = 0, detuning = 0 later on. 
        detuning_volt_vect= np.linspace(-V_range/2+T_v*(bin_size/2-step_size/2),V_range/2-T_v*(bin_size/2+step_size/2),max_it+1)
    
        detuning_summed = [0]
        #Calculate detuning
        for i in range(0, max_it):
            detuning_summed.append(detuning_summed[i] + local_FSRs[i]*step_size*T_v*MZ_calibr)   #MHz. MZ calibrated to detune 98.5 MHz per cycle
        np_detuning_volt_vec = np.asarray(detuning_volt_vect)
        np_detuning_vec = np.asarray(detuning_summed)
        np_detuning_vec_shifted = np_detuning_vec - np_detuning_vec[round(len(np_detuning_vec)/2)] #set middle-data point to detuning = 0
        return np_detuning_volt_vec, np_detuning_vec_shifted

    @staticmethod
    def get_one_sided_PSD(data, n):
        data_fft = np.fft.fft(data,n)
        PSD_2 = abs(data_fft/n)      #doublesided PSD
        PSD_1 = PSD_2[0:int(n/2)]
        PSD_1[1:] = 2*PSD_1[1:]
        return PSD_1

    @staticmethod
    def expand_matrix(matrix, rep_n_times = 4): #pools the size of your data by an amount of twice the floored value of the box size
        expanded = np.repeat(matrix, repeats = int(rep_n_times))
        return expanded

    @staticmethod
    def from_volt_to_steps(to_convert, sampling_rate): # SAMPLING RATE IN V^-1, NOT THE NIDAQSAMPLING RATE
        step_size = to_convert * sampling_rate
        if not step_size.is_integer():
            print("be careful, conversion from volts to steps is not an integer, which may or may not slightly distort all calibrations")
        steps = int(step_size)
        return steps

    @staticmethod
    def run_piezo_sweep_and_wait_until_done(task_ai, task_ao):
        task_ai.start()
        task_ao.start()
        task_ai.wait_until_done()
        task_ao.wait_until_done()

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else 2**math.ceil(math.log2(x))

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
