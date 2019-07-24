import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
from LASERCONTROL.Tl6800control import TL6800
from GUI.yellow_coarse_scan_design import Ui_Form
from LASERCONTROL.Tl6800control import TL6800
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import AnalogSingleChannelReader, AnalogMultiChannelReader
import nidaqmx
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSignal
from DRIVERS.lowpassfilter import butter_lowpass_filter
import threading

import pyqtgraph as pg

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import visa 
from math import ceil, floor
import operator


sys.path.append("Z:\Lab\Pulsing_Setup\python")

# from laser_control.helper_functions_2_0 import vatten_cw


class coarse_scan_GUI(QtGui.QWidget):
    # class variable; laser can now be called from within functions. Connection is made once the GUI is started

    def __init__(self):
        """Create the coarseScan GUI as used in the scanner app. Uses the coarse_scan_design.py from the \drivers folder.
        The actual scanning functionality is set in the scanning classes. These should be callable without the GUI"""
        QtGui.QWidget.__init__(self, None)

        # Set up the user interface from Designer.
        self.ui = Ui_Form()
        self.cs = CoarseScan()
        self.ui.setupUi(self)
        pg.ValueLabel(siPrefix = False, formatStr = None)
        self.plot_wdg = pg.PlotWidget()
        self.plot_wdg.enableAutoRange()

        self.plot_wdg.showAxis('top', show=True)
        self.plot_wdg.showAxis('right', show=True)
        pg.ValueLabel(siPrefix = False, formatStr = None)
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
   
        self.ui.gridLayout.addWidget(self.plot_wdg, 0, 2, 8, 1)
        self.region = pg.LinearRegionItem()
        
        # Connect all the buttons etc
        # scanning
        self.ui.btn_scan.clicked.connect(self.scan) 
        
        self.ui.comboBox_detection.activated[str].connect((lambda: self.switch_channel()))
        self.ui.comboBox_wavelength.activated[str].connect((lambda: self.switch_channel_2()))
        self.ui.comboBox_test_detection.activated[str].connect((lambda: self.switch_channel_3()))
        self.ui.comboBox_pd_no_disk.activated[str].connect((lambda: self.switch_channel_4()))
        self.ui.comboBox_plotting_coarse_scan.activated[str].connect((lambda: self.plot_results()))
        self.ui.btn_save.clicked.connect(self.save_file)
        
        # self.ui.btn_cont.clicked.connect(self.cs_on_off)


    def scan(self):
        """Start the main scanning routine here. Performs single scan over the selected wavelength range"""
        wl_start = float(self.ui.edit_wl_start.text())
        wl_stop = float(self.ui.edit_wl_stop.text())
        fwd_velocity = float(self.ui.lineedit_scanning_speed.text())
        start_lim = 1520
        stop_lim = 1570
        try:
            assert wl_start>=start_lim and wl_start <= stop_lim 
            assert (wl_stop >= start_lim) and (wl_stop <= stop_lim)
            assert wl_stop > wl_start
            assert fwd_velocity <= 20
            assert len(self.cs.list_with_ai_channels) == len(set(self.cs.list_with_ai_channels))
            # ok, so we are good to scan. Pass the values to the routine
            print('Scan from {} to {} nm: '.format(str(wl_start),str(wl_stop)))
            print("Scan with speed {} nm/s".format(fwd_velocity))
            self.cs.wavelength_start = wl_start
            self.cs.wavelength_stop = wl_stop
            self.cs.speed = fwd_velocity
            print(f"Ok then, let's scan the TL6800")
            self.cs.scan_TL6800()
            self.plot_results()
        except AssertionError:
            print("One of your Coarse Scan line edit values is out of range/erronous, or you have defined multiple reading channels on the same cDAQ ai channel.")
            pass
              
    def plot_results(self):
        self.plot_comboBox_text = self.ui.comboBox_plotting_coarse_scan.currentText()
        try:
            if(self.plot_comboBox_text == "Tapered fiber vs wl"):
                new_x_2_disk, new_y_2_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[0])
                new_x_disk_m = self.cs.from_nm_to_m(new_x_2_disk)
                self.plot_wdg.plot(new_x_2_disk, new_y_2_disk, pen = 'w')
                self.visualise_pyqtgraph(new_x_disk_m, new_y_2_disk, xlabel = "Wavelength", ylabel = "Voltage")
                self.show_data(self.cs.xdata, self.cs.ydata[0])  
                print("Plot disk power")
            
            if(self.plot_comboBox_text == "Reference vs wl"):
                print("Plot reference pre-disk power")
                new_x_2_disk, new_y_2_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[2])
                self.show_data(new_x_2_disk, new_y_2_disk)
                self.visualise_pyqtgraph(new_x_2_disk, new_y_2_disk, xlabel = "Wavelength",ylabel = "Voltage", y_units = "V")

            
            if(self.plot_comboBox_text == "Tap/ref"):
                print("Plot normalised power")
                new_x_2_disk, new_y_2_no_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[2])
                new_x_2_disk, new_y_2_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[0])

                y_tapered_fiber = new_y_2_disk/new_y_2_disk[0]
                new_y_2_no_disk_normed = new_y_2_no_disk/new_y_2_no_disk[0]

                fs = self.cs.samples_per_channel/(new_x_2_disk[len(new_x_2_disk)-1]-new_x_2_disk[0])            #sampling freq in terms of nm^-1, in the order of 10^3
                fpass = 50                          

                y_reference_filtered = butter_lowpass_filter(new_y_2_no_disk_normed,fpass,fs)
        
                self.cs.x_data_normalised = new_x_2_disk
                self.cs.ydata_normalised_processed = y_tapered_fiber - y_reference_filtered + 1
                self.show_data(self.cs.x_data_normalised, self.cs.ydata_normalised_processed)
                self.visualise_pyqtgraph(self.cs.x_data_normalised[200:], self.cs.ydata_normalised_processed[200:], xlabel = "Wavelength", ylabel = "Norm'ed transmission")



            if(self.plot_comboBox_text == "pre filter"):
                print("Plot normalised power")
                new_x_2_disk, new_y_2_no_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[2])
                new_x_2_disk, new_y_2_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[0])

                y_tapered_fiber = new_y_2_disk/new_y_2_disk[0]
                new_y_2_no_disk_normed = new_y_2_no_disk/new_y_2_no_disk[0]
                print(new_y_2_no_disk_normed[0:200])
                fs = self.cs.samples_per_channel/(new_x_2_disk[len(new_x_2_disk)-1]-new_x_2_disk[0])            #sampling freq in terms of nm^-1
                fpass = 50

                y_reference_filtered = butter_lowpass_filter(new_y_2_no_disk_normed,fpass,fs)
        
                print(y_reference_filtered[0:200])
                self.visualise_pyqtgraph(new_x_2_disk, new_y_2_no_disk_normed, xlabel = "Wavelength", ylabel = "Norm'ed transmission")

            if(self.plot_comboBox_text == "Test norm"):
                print("Plot normalised power")
                new_x_2_disk, new_y_2_no_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[2])
                new_x_2_disk, new_y_2_disk = self.lin_and_avg(self.cs.xdata, self.cs.ydata[0])

                y_tapered_fiber = new_y_2_disk/new_y_2_disk[0]
                new_y_2_no_disk_normed = new_y_2_no_disk/new_y_2_no_disk[0]
        
                fs = self.cs.samples_per_channel/(new_x_2_disk[len(new_x_2_disk)-1]-new_x_2_disk[0])            #sampling freq in terms of nm^-1
                fpass = 50

                self.y_reference_filtered = butter_lowpass_filter(new_y_2_no_disk_normed,fpass,fs)
            
        
                self.visualise_pyqtgraph(new_x_2_disk,  self.y_reference_filtered, xlabel = "Wavelength", ylabel = "Norm'ed transmission")
        except AttributeError:
                print("Attribute error caught. Maybe you don't have (coarse scan) data yet.")

 
    def lin_and_avg(self, xdata, ydata):
        avg_steps = int(self.ui.lineEdit_avg_over_x_steps.text())
        new_x, new_y = self.cs.average_your_y_data(xdata, ydata, box_size= avg_steps)
        new_x_2, new_y_2 = self.cs.linearize_your_x_data(new_x, new_y, self.cs.wavelength_start, self.cs.wavelength_stop)
        return new_x_2, new_y_2
        
    def visualise_pyqtgraph(self, xdata, ydata, xlabel = None, x_units = None, ylabel = None, y_units = None):
        self.plot_wdg.clear()
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
        self.plot_wdg.enableAutoRange()
        pg.ValueLabel(siPrefix = False)
        pg.ValueLabel(parent = self.plot_wdg, siPrefix = False)
        self.plot_wdg.plot(xdata, ydata, pen = 'w')
        self.plot_wdg.setLabel("bottom",text = str(xlabel), units = str(x_units), unitPrefix = " ")
        self.plot_wdg.setLabel("left", text = str(ylabel), units = str(y_units), unitPrefix = " ")
        pg.ValueLabel(parent = self.plot_wdg, siPrefix = False)
        pg.ValueLabel(siPrefix = False)

    def show_data(self, xdata, ydata):
        self.ui.info_text.clear()
        self.ui.info_text.append("Wl    V \n")
        [self.ui.info_text.append("{}  {}".format(str(xdata[i])[0:4], str(ydata[i])[0:4])) for i in range(0,len(xdata))]


    def save_file(self):
        #Make all this cleaner later on. Include the normalised and filtered data etc.
        self.plot_comboBox_text = self.ui.comboBox_plotting_coarse_scan.currentText()
        
        new_x_2, new_y_2_tapered = self.lin_and_avg(self.cs.xdata, self.cs.ydata[0])
        new_x_2, new_y_2_ref= self.lin_and_avg(self.cs.xdata, self.cs.ydata[2])
      
        try:
            self.write_to_file(new_x_2, new_y_2_tapered, new_y_2_ref, self.cs.x_data_normalised, self.cs.ydata_normalised_processed)
        
        except FileNotFoundError:
            print("Chosen to not save data")
            pass
        except AttributeError:
            print("Attribute error caught. Maybe you don't have (coarse scan) data yet.")
            pass

    def write_to_file(self, xdata, ydata_1, ydata_2, ydata_3, ydata_4):
        try:
            name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat)")
            with open(str(name[0]),'w') as f1:
                f1.writelines("wavelength(nm), Tapered transmission (V), Reference transmission (V), Xdata Normed, Ydata normed (V)  \n") 
                [f1.writelines("{}, {}, {}, {}, {} \n".format(str(xdata[i]), str(ydata_1[i]), str(ydata_2[i]), str(ydata_3[i]), str(ydata_4[i]))) for i in range(0,len(xdata))]
                print("Succesfully wrote {}  data to {}".format(self.plot_comboBox_text, name[0]))
        except FileNotFoundError:
            raise FileNotFoundError
        except AssertionError:
            print("Make sure to plot Tap/Ref first")
            


    def switch_channel(self):                                       #PD_disk detection
        text = self.ui.comboBox_detection.currentText()
        self.cs.channel = "cDAQ1Mod3/" + text                     # = Xdata 
        self.cs.update_channels()
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading PD_disk data at channel ' + self.cs.channel)
                    
    def switch_channel_2(self):                                     #wavelength detection
        text = self.ui.comboBox_wavelength.currentText()
        self.cs.channel_2 = "cDAQ1Mod3/" + text                     # = ydata
        self.cs.update_channels()
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading wavelength data at channel ' + self.cs.channel_2)

    def switch_channel_3(self):                                     #test detection
        text = self.ui.comboBox_test_detection.currentText()
        self.cs.channel_3 = "cDAQ1Mod3/" + text                     # = ydata
        self.cs.update_channels()
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading test data at channel ' + self.cs.channel_3)

    def switch_channel_4(self):                                     #PD_no disk detection
        text = self.ui.comboBox_pd_no_disk.currentText()
        self.cs.channel_4 = "cDAQ1Mod3/" + text
        print('now reading no-disk data at channel ' + self.cs.channel_4)
        self.cs.update_channels()

    def cs_on_off(self):
        if self.stop_flag == False:
            self.stop_flag = True
            print(f'stop cont scan {self.scanning_flag}')
            while True:
                if self.scanning_flag == False:
                    break
                time.sleep(0.1)
                
            self.cont_cs.join()

        elif self.stop_flag == True:
            print("start cont scan")
            # need to set it up again?
            self.stop_flag = False
            self.cont_cs = threading.Thread(None, self.continuous_scan, None)
            self.cont_cs.start()
            # so we are not scanning. Let's start it then

    def continuous_scan(self):
        """Continuous scanning routine for the Santec lasers. It is slow. Maybe try to use the option to 
        scan both ways. But rather use the vidia laser insead
        The scanning is performed in a QThread to allow the main GUI to be accessible all the time. Scanning can then 
        be stopped by pressing the button again. The plotting is also done from the main tread. The worker threads emits
        a QSignal everytime a new trace is available."""
        print("starting cont")
        #The stop_flag will be used to switch the continuous scanning on and off
        #use it here to see if the scan is currently running (False == not running) and decide on action
        
        while True:
            if self.stop_flag == False:
                self.scanning_flag = True        
                print("flags checked")
                #so we are not scanning. Let's start it then
                               
                self.scan()
            else:
                self.scanning_flag = False
                break


class CoarseScan(TL6800):
    """Executes a coarse scan of the laser over a selected range between
    1520 nm to 1670 nm. Two analog inputs are read from the DAQ card. Currently
    only one photodiode is applied. Second input could be used to calibrate the
    laser wavelength with the analog output of the laser."""

    #laser = TL6800()

    def __init__(self):
        """Initialization of the GUI"""
        super().__init__()        #This transfers all instance variables and class functions from the TL6800 class to the coarsescan class)
        self.speed = 20
        self.samples_per_channel = 10000    
        self.channel = 'cDAQ1Mod3/ai0'      #X_data, wavelength detection
        self.channel_2 = "cDAQ1Mod3/ai0"    #Y_data_0, tapered fiber detection
        self.channel_3 = "cDAQ1Mod3/ai0"    #Y_data_1, reference data
        self.channel_4 = "cDAQ1Mod3/ai0"    #Y_data_2, test detection (can be used if needed)
        self.list_with_ai_channels = [self.channel, self.channel_2, self.channel_3, self.channel_4]
        self.number_of_channels = len(self.list_with_ai_channels)
        self.number_of_pds = self.number_of_channels - 1  #number of photodiode detectors responsible for y-data

    def set_scan_params(self):
        """Set the scanning parameters which are dependant on the chosen wavelength range. This function
        has to be called everytime a new range is selected"""
      
        self.sweeping_time = (self.wavelength_stop - self.wavelength_start) / self.speed + 0.1 # 0.1 seems to work. Set to 1 for small scanning range-testing
        self.sampling_rate = self.samples_per_channel / (self.sweeping_time) #increment with time to give the laser some time to process the start of the sweep
        self.wavelength = np.linspace(self.wavelength_start, self.wavelength_stop, self.samples_per_channel)
        self.ydata = np.zeros([self.number_of_pds, self.samples_per_channel])
        print(f'Sweeping time: {self.sweeping_time}')

    def update_channels(self):
        self.list_with_ai_channels = [self.channel, self.channel_2, self.channel_3, self.channel_4]
        
    def scan_TL6800(self):
        """Main Scanning routine"""

    
        self.wavelength_actual = float(self.TL6800_query_wavelength(printbool = False))  
        self.move_to_wavelength_start()                                                     #Have the laser move to wavelength_start
        self.set_scan_params()                                                      #Set variables needed to have the scan run as required (speed values, time required, amount of samples etc.)
        self.set_laser_scan_params()                                                # Pass some of those values to the laser (scan speed, wavelenght start etc.)
        self.perform_coarse_scan()                                                  #Have the laser perform the set of tasks to perform the scan
        print(f'successfully read {len(self.values_read)} samples') 
        self.process_coarse_scan_data()                                               #process cDAQ data
    

    def set_laser_scan_params(self):
        # Make sure that all default//tweekable parameters are set correctly.
        self.TL6800_sweep_parameters(printbool = False)
        self.TL6800_sweep_set_forwardvelocity(fwdvel = self.speed, printbool = False)
        self.TL6800_sweep_set_backwardvelocity(bckwdvel = 20, printbool = False) #backwards can be quick. Don't care about power accuracy whilst not measuring anyway
      
    def process_coarse_scan_data(self):
        # Convert data from volt to nm, store the read cDAQ data to some more convenient names.
        xdata = np.asarray(self.values_read[0])
        xdata_nm = xdata * 5 + 1520        # convert form [V_laser] to [nm]
        self.xdata = xdata_nm
        self.ydata[0] = self.values_read[1]        #pd_tapered_ data
        self.ydata[1] = self.values_read[2]        #test detection
        self.ydata[2] = self.values_read[3]        #reference detection
  
    def perform_coarse_scan(self):
        read_task = nidaqmx.Task()
        self.add_ai_channels_to_task(read_task)   
        self.values_read = np.zeros((self.number_of_channels, self.samples_per_channel), dtype=np.float64) #create buffer to store samples in (this is required if samples>1000)
        reader = AnalogMultiChannelReader(read_task.in_stream)
        print('Start sweeping now.')
        self.TL6800_sweep_start(printbool = False, readbool = False)  # start sweeping
        read_task.start()
        read_task.wait_until_done(timeout = 4000)
        reader.read_many_sample(self.values_read, number_of_samples_per_channel=self.samples_per_channel, timeout=4000)
        read_task.close()
        self.TL6800_read_ascii() #readbool is flagged down in the sweep, now laser buffer has to be emptied

    def add_ai_channels_to_task(self, task):
        print(self.list_with_ai_channels)
        for channels in self.list_with_ai_channels:
            task.ai_channels.add_ai_voltage_chan(str(channels), min_val=-10, max_val=10) 
        task.timing.cfg_samp_clk_timing(rate = self.sampling_rate, samps_per_chan = self.samples_per_channel)

    def move_to_wavelength_start(self):
        #Have the laser move to a certain wavelength. 
        current_wavelength = float(self.TL6800_query_wavelength(printbool = False))
        time_needed = abs((current_wavelength- self.wavelength_start)+2)
        self.TL6800_set_wavelength(self.wavelength_start, printbool = False)
        time.sleep(time_needed)

        #Wait for the laser to stabilize before starting the sweep, for accuracy.
        while True:
            trackmode = self.check_trackmode()
            if not trackmode: break
            
    def check_trackmode(self):
        time.sleep(0.1)
        trackmode = self.TL6800_query_trackmode(printbool = False)
        print("trackmode tested")
        return trackmode

    def save_data(self):
        with open('xdata.csv', 'w+') as f1:
            f1.writelines("%s\n" % points for points in self.values_read[0]) #store uncoverted data
            f1.close()

        with open('ydata.csv', 'w+') as f2:
            f2.writelines("%s\n" % points for points in self.ydata[0])
            f2.close()

    def linearize_your_x_data(self, xdata, ydata, min_wavelength, max_wavelength):
        #TL6800 does not send out a scanning trigger, so there is no real way to know when exacrly it's scanning.
        #Fit a straight line from the point where voltage is a few nm above the min value (to reduce noise), and a few nm below the max value. Find all indices of these x-values, make a 
        # linspace between min wl, max wl, with nr of found x-values, take a slice at y of that amount, plot the two
        #of the y-values, and plt those to
        index, value = max(enumerate(self.xdata), key = operator.itemgetter(1))
        #indices([1,0,3,5,1], lambda x: x==1) from the internetz
        treshold = 0.15
        some_indices = self.indices(xdata, lambda x:( x>(min_wavelength + treshold)))
        some_more_indices = self.indices(xdata, lambda x:( x<(max_wavelength - treshold)))
        indices = list(set(some_indices).intersection(some_more_indices))
        min_index = min(indices)
        max_index = max(indices)
        #min_wavelength_2 = xdata[min_index]
        #max_wavelength_2 = xdata[max_index]
        x_new= np.linspace(min_wavelength+treshold,max_wavelength - treshold, max_index - min_index)
        y_new = ydata[min_index:max_index]
        assert len(x_new) == len(y_new)
        return x_new, y_new
        
    @staticmethod    
    def indices(list, filtr=lambda x: bool(x)): #get indices that meet a certain condition set by bool(x).
        all_indices = [i for i,x in enumerate(list) if filtr(x)]
        return all_indices 

    @staticmethod
    def average_your_y_data(x_data,y_data, box_size = 5): #pools the size of your data by an amount of twice the floored value of the box size
        average_x_points = int(box_size)
        len_xdata = len(x_data)
        len_ydata = len(y_data)

        floor_a_x_p = floor(average_x_points/2)
        y_avgd = np.zeros(len_ydata-2*floor_a_x_p)
        x_new = x_data[floor_a_x_p:(len_xdata - floor_a_x_p)]
        
        for i in range(floor_a_x_p, len_xdata-floor_a_x_p):
            left_bound = i - floor_a_x_p
            right_bound = i + floor_a_x_p
            y_avgd[i-floor_a_x_p] = np.mean(y_data[left_bound: right_bound])

        return x_new, y_avgd
    
    @staticmethod
    def from_nm_to_m(data):
        data_converted = data * 10**-9
        return data_converted

    @staticmethod
    def from_volt_to_steps(to_convert):
        step_size = to_convert
        if not step_size.is_integer():
            print("be careful, conversion from volts to steps is not an integer, which may or may not slightly distort all calibrations")
        steps = int(step_size)
        return steps
            
if __name__ == '__main__':

    app = QtGui.QApplication(sys.argv)
    myWidget = QtGui.QMainWindow()

    # main layout
    mainLayout = QtGui.QVBoxLayout()

    # add all main to the main vLayout
    mainLayout.addWidget(coarse_scan_GUI())
    # self.mainLayout.addWidget(self.scrollArea)

    # central widget
    centralWidget = QtGui.QWidget()
    centralWidget.setLayout(mainLayout)

    # set central widget
    myWidget.setCentralWidget(centralWidget)
    myWidget.show()
    app.exec_()