from own_coarse_scan_design import Ui_Form
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
from math import ceil, floor

sys.path.append("Z:\Lab\Pulsing_Setup\python")

# from laser_control.helper_functions_2_0 import vatten_cw


class coarseScanGUI(QtGui.QWidget):
    # class variable; laser can now be called from within functions. Connection is made once the GUI is started

    def __init__(self):
        """Create the coarseScan GUI as used in the scanner app. Uses the coarse_scan_design.py from the \drivers folder.
        The actual scanning functionality is set in the scanning classes. These should be callable without the GUI"""

        QtGui.QWidget.__init__(self, None)

        # Set up the user interface from Designer.
        self.ui = Ui_Form()
        self.cs = CoarseScan()
        self.ui.setupUi(self)

        self.plot_wdg = pg.PlotWidget()
        self.plot_wdg.enableAutoRange()

        self.plot_wdg.showAxis('top', show=True)
        self.plot_wdg.showAxis('right', show=True)

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
        self.ui.comboBox_plotting_coarse_scan.activated[str].connect((lambda: self.plot_results()))
        self.ui.btn_save.clicked.connect(self.save_file)
        
        # self.ui.btn_cont.clicked.connect(self.cs_on_off)


    def scan(self):
        """Start the main scanning routine here. Performs single scan over the selected wavelength range"""

        # Select the wavelength range for scanning
        wl_start = float(self.ui.edit_wl_start.text())
        wl_stop = float(self.ui.edit_wl_stop.text())
        fwd_velocity = float(self.ui.lineedit_scanning_speed.text())
        print('input range: ' + str(wl_start) + ' - ' + str(wl_stop))
        start_lim = 1520
        stop_lim = 1570

        if (wl_start >= start_lim) and (wl_start <= stop_lim):  # laser limits
            if (wl_stop >= start_lim) and (wl_stop <= stop_lim):  # laser limits
                if wl_stop > wl_start:

                    # ok, so we are good to scan. Pass the values to the routine
                    self.cs.wavelength_start = wl_start
                    self.cs.wavelength_stop = wl_stop
                    self.cs.speed = fwd_velocity
 
                    # update the scanning parameters to the new wavelength range
                    # self.cs.set_params()

                    print(f"Ok then, let's scan the TL6800")
                    self.cs.scan_TL6800()
                    print("baak in the main routine")
               
                    self.plot_wdg.clear()
                    print("cleared")
                    self.plot_wdg.enableAutoRange()
           
                    self.plot_results()
              
    def plot_results(self):
        self.plot_comboBox_text = self.ui.comboBox_plotting_coarse_scan.currentText()
        
        self.plot_wdg.clear()
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
    
        #self.plot_wdg.addItem(self.region)

        self.plot_wdg.enableAutoRange()
        

        try:
            if(self.plot_comboBox_text == "Disk_power vs wl"):
                print("Plot disk power")
                print("size input x", np.size(self.cs.xdata))
                print("size input y", np.size(self.cs.ydata[0]))
                avg_steps = int(self.ui.lineEdit_avg_over_x_steps.text())
                if avg_steps > 1.5:
                    print("averaging")
                    new_x_disk, new_y_disk = self.cs.average_your_y_data(self.cs.xdata, self.cs.ydata[0], box_size= avg_steps, box_in_volts = False)
                    self.show_data(new_x_disk, new_y_disk)
                    self.plot_wdg.plot(new_x_disk, new_y_disk, pen = 'w')
                else: 
                    print("not averaging")
                    self.show_data(self.cs.xdata, self.cs.ydata[0])
                    self.plot_wdg.plot(self.cs.xdata, self.cs.ydata[0], pen = 'w')
                #self.show_data(self.cs.xdata, self.cs.ydata[0])
                #self.plot_wdg.plot(self.cs.xdata, self.cs.ydata[0], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Wavelength", units = None)
                self.plot_wdg.setLabel("left", text = "Wavelength", units = None)
                

            if(self.plot_comboBox_text == "Nat_power vs wl"):
                print("Plot natural pre-disk power")
                self.show_data(self.cs.xdata, self.cs.ydata[1])
                self.plot_wdg.plot(self.cs.xdata, self.cs.ydata[1], pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Wavelength", units = None)
                self.plot_wdg.setLabel("left", text = "Voltage", units = "V")
                
            
            if(self.plot_comboBox_text == "Disk_power/Nat_power"):
                print("Plot normalised power")
                self.show_data(self.cs.xdata, self.cs.ydata_3)
                self.cs.ydata_3 = self.cs.ydata[1]/self.cs.ydata[0]
                self.plot_wdg.plot(self.cs.xdata, self.cs.ydata_3, pen = 'w')
                self.plot_wdg.setLabel("bottom",text = "Wavelength", units = None)
                self.plot_wdg.setLabel("left", text = "Voltage", units = "V")
            
        except:
            print("Plotting failed. Possibly you don't have data yet..")
     

    def show_data(self, xdata, ydata):
        self.ui.info_text.clear()
        self.ui.info_text.append("Wl    V \n")
        [self.ui.info_text.append("{}  {}".format(str(xdata[i])[0:4], str(ydata[i])[0:4])) for i in range(0,len(xdata))]

    def save_file(self):
        self.plot_comboBox_text = self.ui.comboBox_plotting_coarse_scan.currentText()
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File', "Data Files (*.txt *.dat)") 
       
        try:
            if(self.plot_comboBox_text == "Disk_power vs wl"):
                f3= open(str(name[0]),'w')
                f3.writelines("Sweep from {} to {} [nm]. Scanning speed: {} [nm/s] \n".format(self.cs.wavelength_start, self.cs.wavelength_stop, self.cs.speed))
                f3.writelines("wavelength(nm)   Photodiode (from disk) output(V) \n")
                [f3.writelines("{} {}\n".format(str(self.cs.xdata[i]), str(self.cs.ydata[0][i]))) for i in range(0,len(self.cs.xdata))]
                f3.close()
                print("Succesfully wrote disk_power vs wl data to {}".format(name[0]))

            if(self.plot_comboBox_text == "Nat_power vs wl"):
                f3= open(str(name[0]),'w')
                f3.writelines("Sweep from {} to {} [nm]. Scanning speed: {} [nm/s] \n".format(self.cs.wavelength_start, self.cs.wavelength_stop, self.cs.speed))
                f3.writelines("wavelength(nm)   Photodiode (from disk) output(V) \n")
                [f3.writelines("{} {}\n".format(str(self.cs.xdata[i]), str(self.cs.ydata[1][i]))) for i in range(0,len(self.cs.xdata))]
                f3.close()
                print("Succesfully wrote disk_power vs wl data to {}".format(name[0]))

            if(self.plot_comboBox_text == "Disk_power/Nat_power"):
                f3= open(str(name[0]),'w')
                f3.writelines("Sweep from {} to {} [nm]. Scanning speed: {} [nm/s] \n".format(self.cs.wavelength_start, self.cs.wavelength_stop, self.cs.speed))
                f3.writelines("wavelength(nm)   Photodiode (from disk) output(V) \n")
                [f3.writelines("{} {}\n".format(str(self.cs.xdata[i]), str(self.cs.ydata_3[i]))) for i in range(0,len(self.cs.xdata))]
                f3.close()
                print("Succesfully wrote disk_power vs wl data to {}".format(name[0]))

            print("did get til here though")
        except FileNotFoundError:
            print("Chosen not to save data")
            pass
       


    def switch_channel(self):
        text = self.ui.comboBox_detection.currentText()
        self.cs.channel = "cDAQ1Mod3/" + text                     # = Xdata 
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading PD_disk data at channel ' + self.cs.channel)
                    
    def switch_channel_2(self):
        text = self.ui.comboBox_wavelength.currentText()
        self.cs.channel_2 = "cDAQ1Mod3/" + text                     # = Xdata 
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading wavelength data at channel ' + self.cs.channel_2)

    def switch_channel_3(self):
        text = self.ui.comboBox_test_detection.currentText()
        self.cs.channel_3 = "cDAQ1Mod3/" + text                     # = Xdata 
        #self.cs.channel = "cDAQ1Mod3/ai2"
        print('now reading wavelength data at channel ' + self.cs.channel_3)

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
        # Variable declaration. Change these numbers to change the scanning range
        self.wavelength_start
        self.wavelength_stop 
        self.speed = 20   
        print(self.wavelength_start)
       

        self.channel = 'cDAQ1Mod3/ai2'      #X_data. 
        self.channel_2 = "cDAQ1Mod3/ai3"    #Y_data
        self.channel_3 = "cDAQ1Mod3/ai4"    #Y_data
        self.list_with_ai_channels = [self.channel, self.channel_2, self.channel_3]
        self.number_of_channels = len(self.list_with_ai_channels)
        self.number_of_pds = self.number_of_channels - 1  #number of photodiode detectors (which are responsible for y-data)
        

    def set_scan_params(self):
        """Set the scanning parameters which are dependant on the chosen wavelength range. This function
        has to be called everytime a new range is selected"""
      
        self.sweeping_time = (self.wavelength_stop - self.wavelength_start) / self.speed + 0.1 # 0.1 seems to work. Set to 1 for small scanning range-testing
        
        self.samples_per_channel = 5000 # 1000 is maximum (found out the hard way)
        self.sampling_rate = self.samples_per_channel / (self.sweeping_time) #increment with time to give the laser some time to process the start of the sweep
        self.wavelength = np.linspace(self.wavelength_start, self.wavelength_stop, self.samples_per_channel)
        self.ydata = np.zeros([self.number_of_pds, self.samples_per_channel])

        print(f'Sweeping time {self.sweeping_time}')
        print(f'Sampling rate {self.sampling_rate} /sec')
        print(f'Number of samples {self.samples_per_channel}')

        # The wavelength is taken as an array of equidistant values spanning the
        # scanning range. This was found to be precise enough to find the
        # resonances later on in fine sweep mode. The wl could, however, also
        # be calibrated by the analog output of the laser
        
        

    def scan_TL6800(self):
        """Main Scanning routine"""
        #connect to laser via Ethernet now!
        self.wavelength_actual = float(self.TL6800_query_wavelength(printbool = False))
        self.move_to_wavelength_start()
        self.set_scan_params()
        self.set_laser_scan_params()
        self.perform_coarse_scan()
        print(f'successfully read {len(self.values_read)} samples')
        self.process_coarse_scan_data()

    def set_laser_scan_params(self):
        self.TL6800_sweep_parameters(printbool = False)
        self.TL6800_sweep_set_forwardvelocity(fwdvel = self.speed, printbool = False)
        self.TL6800_sweep_set_backwardvelocity(bckwdvel = 20, printbool = False) #backwards can be quick. Don't care about power accuracy whilst not measuring anyway
      
    def process_coarse_scan_data(self):    
        xdata = np.asarray(self.values_read[0])
        xdata_nm = xdata * 5 + 1520        # convert form [V_laser] to [nm]
        self.xdata = xdata_nm
        self.ydata[0] = self.values_read[1]
        self.ydata[1] = self.values_read[2]
  

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
        for channels in self.list_with_ai_channels:
            task.ai_channels.add_ai_voltage_chan(str(channels), min_val=-10, max_val=10) 
        task.timing.cfg_samp_clk_timing(rate = self.sampling_rate, samps_per_chan = self.samples_per_channel)

    def move_to_wavelength_start(self):
        current_wavelength = float(self.TL6800_query_wavelength(printbool = False))
        time_needed = abs((current_wavelength- self.wavelength_start)+2)
        self.TL6800_set_wavelength(self.wavelength_start, printbool = False)
        time.sleep(time_needed)

        #Laser is unstable during trackmode. Trackmode turns off some time after reaching the entered wavelength.  Have the finesweep
        #wait until trackmode is off (-> set wavelength == current wavelength) before entering the finesweep
        while True:
            trackmode = self.check_trackmode()
            if not trackmode: break
            
    def check_trackmode(self):
        time.sleep(0.2)
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
    

    def average_your_y_data(self, x_data,y_data, box_size = 5, box_in_volts = False): #pools the size of your data by an amount of twice the floored value of the box size
        if(box_in_volts == True):
            average_x_points = self.from_volt_to_steps(box_size, self.sampling_rate) #doesnt work yet. Work on this later
        else:
            average_x_points = int(box_size)
            pass
        
        len_xdata = len(x_data)
        len_ydata = len(y_data)
        assert len_xdata == len_ydata
        floor_a_x_p = floor(average_x_points/2)
        y_avgd = np.zeros(len_ydata-2*floor_a_x_p)
        x_new = x_data[floor_a_x_p:(len_xdata - floor_a_x_p)]
        
        for i in range(floor_a_x_p, len_xdata-floor_a_x_p):
            left_bound = i - floor_a_x_p
            right_bound = i + floor_a_x_p
            y_avgd[i-floor_a_x_p] = np.mean(y_data[left_bound: right_bound])
        assert np.size(x_new) == np.size(y_avgd)

        return x_new, y_avgd

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
    mainLayout.addWidget(coarseScanGUI())
    # self.mainLayout.addWidget(self.scrollArea)

    # central widget
    centralWidget = QtGui.QWidget()
    centralWidget.setLayout(mainLayout)

    # set central widget
    myWidget.setCentralWidget(centralWidget)


    myWidget.show()
    app.exec_()
