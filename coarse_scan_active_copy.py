# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:29:34 2016

@author: Andreas
"""
from PyQt4 import QtGui, QtCore


import pyqtgraph as pg
import numpy as np
import time
import sys

from save_traces import saveTrace
from drivers.lorentzfit import *
from drivers.SantecDrivers import SantecTSL510
from drivers.DAQmx_classes import *
from nidaqmx import AnalogInputTask

from drivers.coarse_scan_design import Ui_Form
from helper_functions2 import vatten_cw
import matplotlib.pyplot as plt



class coarseScanGUI(QtGui.QWidget):
    def __init__(self):
        """Create the coarseScan GUI as used in the scanner app. Uses the coarse_scan_design.py from the \drivers folder.
        The actual scanning functionality is set in the scanning classes. These should be callable without the GUI"""

        QtGui.QWidget.__init__(self, None)

        # Set up the user interface from Designer.
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        #set up the pyqtgraph in here. This cannot be created in the QtDesigner
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')

        self.plot_wdg = pg.PlotWidget()
        self.plot_wdg.enableAutoRange()

        self.plot_wdg.showAxis('top', show=True)
        self.plot_wdg.showAxis('right', show=True)

        self.ui.gridLayout.addWidget(self.plot_wdg, 0, 2, 8, 1)
        
        #Connect all the buttons etc
        #scanning
        self.ui.btn_scan.clicked.connect(self.scan)
        self.ui.btn_cont.clicked.connect(self.continuousScan)
        #select the photodiodes
        self.ui.radio_1.toggle()
        self.connect(self.ui.radio_1, QtCore.SIGNAL("clicked()"), lambda: self.switch_channel('0'))
        self.connect(self.ui.radio_2, QtCore.SIGNAL("clicked()"), lambda: self.switch_channel('1'))
        self.connect(self.ui.radio_3, QtCore.SIGNAL("clicked()"), lambda: self.switch_channel('2'))
        #saving
        self.ui.btn_save.clicked.connect(self.save)
        self.ui.btn_save_txt.clicked.connect(self.save_txt)
        #VOA -- put this somewhere else maybe?
        #self.ui.btn_vatten.clicked.connect(self.set_vatten)
        
        #connect to the actual scanning routines
        self.cs = CoarseScan()
        self.cont_cs = contScan()
        self.connect(self.cont_cs, self.cont_cs.signal, self.update_plot)    #connect the Qtsignal to the Gui, see update_plot



    def set_vatten(self):
        """set the attenuator voltage. Currently WIP. Also probably not the right place to put this"""

        #if self.on_off == 0:
        try:
            set_pwr = float(self.ui.edit_vattn_2.text())


            if (set_pwr >= 0.00) and (set_pwr <= 5):
                    attenuator = vatten_cw()
                    voltage = set_pwr #attenuator.get_voltage(set_pwr)
                    #np.round([0.37, 1.64], decimals=2)
                    voltage = np.round(voltage, decimals=3)
                    print('setting vatten to ' + str(voltage))
                    attenuator.write_voltage(voltage)
        except:
            self.ui.edit_vattn_2.setText('wrong input')



    def switch_channel(self, channel):
                
        self.cs.channel = channel
        self.cont_cs.channel = channel
        print('now using ai' + self.cs.channel)
        

    def save_txt(self):
        """Save the data shown in the Coarse Scan window in a text format"""
        name = QtGui.QFileDialog.getSaveFileName(self, "Save File", '')
        print name
        wavelength = self.cs.wavelength
        ydata = self.cs.ydata
        np.savetxt(str(name), np.append([wavelength], [ydata], axis=0).T, delimiter='\t')
        
        
    def save(self):
        """Save the data shown in the Coarse Scan window in the recent scans database. Still WIP."""
        
        #take the text from the lineedit to save as some info about the trace
        info = str(self.ui.info_text.toPlainText())
        if info == '':
            info = 'no info given'
        print info

        #open our usual database. The functionality for this is in the respective lib
        db = saveTrace()
        db.openTodaysDB(autocreate=1)
        #pass all the data we want to store
        db.write(self.cs.wavelength, self.cs.ydata, 'coarse_scan', info)
        #and we are done
        db.close()

    def scan(self):
        """Start the main scanning routine here. Performs single scan over the selected wavelength range"""

        try:
            #Select the wavelength range for scanning
            wl_start = float(self.ui.edit_wl_start.text())
            wl_stop = float(self.ui.edit_wl_stop.text())
            
            print('input range: ' + str(wl_start) + ' - ' + str(wl_stop))
            if (wl_start >= 1500.00) and (wl_start <= 1630.00):  # laser limits
                if (wl_stop >= 1500.00) and (wl_stop <= 1630.00):  # laser limits
                    if wl_stop > wl_start:
                        #ok, so we are good to scan. Pass the values to the routine
                        self.cs.wavelength_start = wl_start
                        self.cs.wavelength_stop = wl_stop
                        
                        #update the scanning parameters to the new wavelength range
                        self.cs.set_params()

                        print("Ok then, let's scan")
                        self.cs.scan()              #run the scan
                        
                        #done with the scanning. Lets plot this
                        self.plot_wdg.clear()
                        self.plot_wdg.enableAutoRange()
                        self.plot_wdg.plot(self.cs.wavelength, self.cs.ydata, pen='w')
        except:
            #assume here the error occurs if the lineinput is wrong. This is prob not good error catching..
            print('Something is wrong with your input')


    def continuousScan(self):
        """Continuous scanning routine for the Santec TSL510. It is slow. Maybe try to use the option to 
        scan both ways. But rather use the vidia laser insead
        The scanning is performed in a QThread to allow the main GUI to be accessible all the time. Scanning can then 
        be stopped by pressing the button again. The plotting is also done from the main tread. The worker threads emits
        a QSignal everytime a new trace is available."""
        
        #The stop_flag will be used to switch the continuous scanning on and off
        #use it here to see if the scan is currently running (False == not running) and decide on action
        if self.cont_cs.stop_flag == 0:
            self.cont_cs.stop_flag = 1

        elif self.cont_cs.stop_flag == 1:
            #so we are not scanning. Let's start it then
            self.cont_cs.stop_flag = 0
            
            #read the desired wavelength from the user input
            wl_start = float(self.ui.edit_wl_start.text())
            wl_stop = float(self.ui.edit_wl_stop.text())
            print('input range: ' + str(wl_start) + ' - ' + str(wl_stop))
            if (wl_start >= 1500.00) and (wl_start <= 1630.00):  # laser limits
                if (wl_stop >= 1500.00) and (wl_stop <= 1630.00):  # laser limits
                    if wl_stop > wl_start:
                        #ok, so we are good to scan. Pass the values to the routine
                        self.cont_cs.wavelength_start = wl_start
                        self.cont_cs.wavelength_stop = wl_stop
                        
                        #update the scanning parameters to the new wavelength range
                        self.cont_cs.set_params()
    
    
                        #start the scanning worker thread and exit the main thread immediately. Plotiing will be done
                        #in update_plot
                        print("Ok then, let's scan")
                        self.cont_cs.start()
                        
                    
    def update_plot(self):
        """update the plot everytime a new trace is available from the continuous_scan.
        The plotting could in principle be done in the worker thread. However that produces a lot of QT warnings.
        This function is called with the QSignal from the worker thread"""
        
        #check if we are actually supposed to plot smth. Redundant
        if self.cont_cs.print_flag  != 0:
            print('triggered, painting now')

            #still not sure if the plot.clear()+replot or the plot.setData() is better
            self.plot_wdg.clear()
            self.plot_wdg.plot(self.cont_cs.wavelength, self.cont_cs.ydata, pen='w')
            
            #Reset the flag so the worker knows we are done. The flags are prob not necessary. Scanning takes much longer than plotting
            self.cont_cs.print_flag = 0


class CoarseScan():
    """Executes a coarse scan of the laser over a selected range between
    1500 nm to 1630 nm. Two analog inputs are read from the DAQ card. Currently
    only one photodiode is applied. Second input could be used to calibrate the
    laser wavelength with the analog output of the laser."""

    def __init__(self):
        """Initialization of the GUI"""

        # Variable declaration. Change these numbers to change the scanning range
        self.wavelength_start = 1500.0  # nm
        self.wavelength_stop = 1630.0  # nm
        self.speed = 50.0  # nm/s

        self.channel = '0'      #used to switch between two possible input channels. Standart is '0' for
                                #reflection but it can be toggled to '1' for transmission
        self.set_params()
        

    def set_params(self):
        """Set the scanning parameters which are dependant on the chosen wavelength range. This function
        has to be called everytime a new range is selected"""

        self.sweeping_time = (self.wavelength_stop - self.wavelength_start) / self.speed  # s
        self.sampling_rate = int(self.speed / 0.005)
        self.samples_per_channel = int(self.sampling_rate * self.sweeping_time)

        # The wavelength is taken as an array of equidistant values spanning the
        # scanning range. This was found to be precise enough to find the
        # resonances later on in fine sweep mode. The wl could, however, also
        # be calibrated by the analog output of the laser
        self.wavelength = np.linspace(self.wavelength_start, self.wavelength_stop, self.samples_per_channel)
        self.ydata = np.zeros(self.samples_per_channel)


    def find_resonances(self):
        """Automatically find the resonances from carse scan. Currently WIP and not used! Try searching for the res with the max of derivation"""
        val = np.argmin(self.ydata)

        mask = np.zeros(self.samples_per_channel)
        mask[val - 8:val + 8] = 1

        fit_xdata = self.wavelength[mask]
        fit_ydata = self.ai0_data[mask]

        intens = np.fabs(np.amax(fit_xdata) - np.amax(fit_xdata))
        p = [0.01, fit_xdata[np.argmin(fit_ydata)], intens, fit_xdata[0]]
        best_parameters = lorentzfit(fit_xdata, fit_ydata, p)

        fit = lorentzian(fit_xdata, best_parameters)
        self.plot_wdg.plot(fit_xdata, fit, pen='g')
        

    def scan(self):
        """Main Scanning routine"""
        #connect to laser via Ethernet now!
        with SantecTSL510("TCPIP0::192.168.1.108::1470::SOCKET") as laser:
            
            # --------------------------------------------------------------------------
            # Create and initialize devices
            # --------------------------------------------------------------------------
            
            input_task = TriggeredInputTask("Dev1/ai" + self.channel, 0, 10,  # Photodetector channel, minimum value, maximum value
                                            "Dev1/ai3", 0, 0.3,  # Laser channel, minimum value, maximum value
                                            "APFI0",  # Trigger channel
                                            self.sampling_rate, self.samples_per_channel)
            print('initializing')
            try:
                # old_wavelength = float(laser.wavelength)   #Save the wavelength to which the laser was
                # set to so we can go back to it once the program is over
                laser.sweep_mode = 1  # set laser to sweeping one-way mode
                laser.wavelength_start = self.wavelength_start  # set sweeping start
                laser.wavelength_stop = self.wavelength_stop  # set sweeping stop
                laser.speed = self.speed  # set sweeping speed
                laser.trigger_output = 'on'  # enable the laser's trigger output
                laser.trigger_timing = 2  # set trigger to the start of a sweep

                # Averaging currently disabled. Put a number > 1 here to use it
                for i in range(1):
                    laser.execute()  # start sweeping
                    input_task.StartTask()  # start task (waits for laser trigger to start acquiring)
                    print('starting the scan')

                    # read data on the DAQ's buffer
                    input_task.read()  

                    # stop task
                    input_task.StopTask()  

                    # photodetector voltage #1::2
                    ai0_data = input_task.data[::2]  

                    # PD voltage vs. wavelength, currently disabled
                    # tmp_wavelength = ai1_data/np.amax(ai1_data) * (wavelength_stop-wavelength_start) + wavelength_start
                    # wavelength = (i*wavelength + tmp_wavelength)/(i+1.) #update the average wavelength array

                    self.ydata = (i * self.ydata + ai0_data) / (i + 1.)  # update the average data to save
                    
                    time.sleep(0.001)

                    while laser.sweep_status != 0:
                        time.sleep(0.05)  # wait until the laser is ready to go

                laser.wavelength = 1550.000  # Set the laser to 1550 nm, can also set it back to the old wl here


            # --------------------------------------------------------------------------
            # Safely close devices
            # --------------------------------------------------------------------------
            finally:
                laser.trigger_output = 'off'
                input_task.ClearTask()  # clear input channels
                
                
                

class contScan(QtCore.QThread):
    """Performs a continuous scanning with the Santec Laser. Does not actually use the built-in 
    continuous function, but starts single scans repetetively for simplicity. The 
    class is built a a QThread object to allow responitivity of the main thread, i.e. the user interface.
    A QSignal is emitted everytime a new trace is available. The scanning is stopped by setting the stop_flag variable."""

    def __init__(self):
        QtCore.QThread.__init__(self)
        
        #Scan variable declaration
        self.wavelength_start = 1500.0  # nm
        self.wavelength_stop = 1630.0  # nm
        self.speed = 50.0  # nm/s
        
        #Two flags to show wether a scanning or plotting action is needed
        self.stop_flag = 1		#True means it is NOT running. Or supposed to stop.
        self.print_flag = 0         #one means there is a new trace available for printing


        self.channel = '0'      #used to dynamically switch between two possible input channels. Standart is '0' for
                                #reflection but it can be toggled to '1' for transmission

        #prepare signal to get main thread to plot new trace
        self.signal = QtCore.SIGNAL("signal")
        
        self.set_params()

    def set_params(self):
        """Set the scanning parameters which are dependant on the chosen wavelength range. This function
        has to be called everytime a new range is selected"""

        self.sweeping_time = (self.wavelength_stop - self.wavelength_start) / self.speed  # s
        self.sampling_rate = int(self.speed / 0.005)
        self.samples_per_channel = int(self.sampling_rate * self.sweeping_time)

        # The wavelength is taken as an array of equidistant values spanning the
        # scanning range. This was found to be precise enough to find the
        # resonances later on in fine sweep mode. The wl could, however, also
        # be calibrated by the analog output of the laser
        self.wavelength = np.linspace(self.wavelength_start, self.wavelength_stop, self.samples_per_channel)
        self.ydata = np.zeros(self.samples_per_channel)
        
        
    def run(self):
        #connect to laser
        with SantecTSL510("TCPIP0::192.168.1.108::1470::SOCKET") as laser:

            #The cont scanning uses the nidaqmx library. At this time it is not perfectly clear wether this works
            #exactly as the old one. There is definitely still some weired stuff going on with some of the values. 
            #also the triggering is probably not very good
            readtask = AnalogInputTask()            
            readtask.create_voltage_channel("Dev1/ai" + self.channel, min_val=0, max_val=10)
            readtask.configure_trigger_analog_edge_start("APFI0")
            readtask.configure_timing_sample_clock(
                                      source = 'OnboardClock', 
                                      rate = self.sampling_rate, # Hz
                                      active_edge = 'rising', 
                                      sample_mode = 'finite', 
                                      samples_per_channel = self.samples_per_channel)

            try:
                laser.sweep_mode = 1  # set laser to sweeping one-way mode.
                laser.wavelength_start = self.wavelength_start  # set sweeping start
                laser.wavelength_stop = self.wavelength_stop  # set sweeping stop
                laser.speed = self.speed  # set sweeping speed
                laser.trigger_output = 'on'  # enable the laser's trigger output
                laser.trigger_timing = 2  # set trigger to the start of a sweep

                #Loop as long as the stop_flag is not set via the GUI
                while self.stop_flag == 0:
                    if self.stop_flag == 1:
                        break
						
                    print('\nstarting next scan')
                    laser.execute()  # start sweeping
                    ai0_data = readtask.read(self.samples_per_channel, timeout=15.0)
                    self.ydata = ai0_data[:,0] # update the average data to save
                    
                    #So we have new data, lets print it then
                    self.print_flag = 1
                    
                    self.emit(self.signal)
                    while self.print_flag != 0:
                        print('waiting to resume')
                        time.sleep(1)  # wait until the laser is ready to go



                    while laser.sweep_status != 0:
                        time.sleep(0.05)  # wait until the laser is ready to go
                        
                    self.stop_flag == 1

                    
                print("...aaand I'm done")

                laser.wavelength = 1550.000  # Set the laser to 1550 nm, can also set it back to the old wl here


            # --------------------------------------------------------------------------
            # Safely close devices
            # --------------------------------------------------------------------------
            finally:
                laser.trigger_output = 'off'




                

class contScanHack():

    def __init__(self):
        """"""

        # Variable declaration
        self.wavelength_start = 1500.0  # nm
        self.wavelength_stop = 1630.0  # nm
        self.speed = 50.0  # nm/s
        self.stop_flag = 1		#True means it is NOT running
        self.print_flag = 0

        self.res_1 = 0
        self.res2 = 0

        self.channel = '0'      #used to dynamically switch between two possible input channels. Standart is '0' for
                                #reflection but it can be toggled to '1' for transmission

        #prepare signal to get main thread to plot new trace
        self.fig = plt.figure()  # Declare figure
        plt.ion()  # Turn on interactive updating
        
        self.ax = self.fig.add_subplot(111)  # add subplot

        self.set_params()

    def set_params(self):

        self.sweeping_time = (self.wavelength_stop - self.wavelength_start) / self.speed  # s
        self.sampling_rate = int(self.speed / 0.005)
        self.samples_per_channel = int(self.sampling_rate * self.sweeping_time)


        # The wavelength is taken as an array of equidistant values spanning the
        # scanning range. This was found to be precise enough to find the
        # resonances later on in fine sweep mode. The wl could, however, also
        # be calibrated by the analog output of the laser
        self.wavelength = np.linspace(self.wavelength_start, self.wavelength_stop, self.samples_per_channel)
        self.ydata = np.zeros(self.samples_per_channel)
        
        
    def scan(self):
        plt.show()  # Show figure

        with SantecTSL510("TCPIP0::192.168.1.108::1470::SOCKET") as laser:

            # --------------------------------------------------------------------------
            # Create and initialize devices
            # --------------------------------------------------------------------------
            #input_task = TriggeredInputTask("Dev1/ai" + self.channel, 0, 0.3,  # Photodetector channel, minimum value, maximum value
            #                                "Dev1/ai2", 0, 0.3,  # Laser channel, minimum value, maximum value
            #                                "APFI0",  # Trigger channel
            #                                self.sampling_rate, self.samples_per_channel)
                                            
            readtask = AnalogInputTask()

            
            readtask.create_voltage_channel("Dev1/ai" + self.channel, min_val=0, max_val=0.5)
            readtask.configure_trigger_analog_edge_start("APFI0")
            #readtask.configure_timing_sample_clock(source='/Dev1/ao/SampleClock',sample_mode = 'continuous',rate = self.sampling_rate) #Setting up sample clock for the input to be equal to that of the output#
            readtask.configure_timing_sample_clock(
                                      source = 'OnboardClock', 
                                      rate = self.sampling_rate, # Hz
                                      active_edge = 'rising', 
                                      sample_mode = 'finite', 
                                      samples_per_channel = self.samples_per_channel)

            try:
                # old_wavelength = float(laser.wavelength)   #Save the wavelength to which the laser was
                # set to so we can go back to it once the program is over
                laser.sweep_mode = 1  # set laser to sweeping one-way mode. THIS SHOULD BE CHANGED
                laser.wavelength_start = self.wavelength_start  # set sweeping start
                laser.wavelength_stop = self.wavelength_stop  # set sweeping stop
                laser.speed = self.speed  # set sweeping speed
                laser.trigger_output = 'on'  # enable the laser's trigger output
                laser.trigger_timing = 2  # set trigger to the start of a sweep

                # Averaging currently disabled
                #input_task.StartTask()  # start task (waits for laser trigger to start acquiring)

                #Loop as long as the stop_flag is not set via the GUI
                while True:
                    try:
                        print('\nstarting next scan')
                        laser.execute()  # start sweeping
                        
                        #input_task.read()  # read data on the DAQ's buffer
                        ai0_data = readtask.read(self.samples_per_channel, timeout=15.0)

                        # PD voltage vs. wavelength, currently disabled
                        # tmp_wavelength = ai1_data/np.amax(ai1_data) * (wavelength_stop-wavelength_start) + wavelength_start
                        # wavelength = (i*wavelength + tmp_wavelength)/(i+1.) #update the average wavelength array

                        self.ydata = ai0_data[:,0] # update the average data to save
                        #mz = ai0_data[:,0]          #MZ


                        self.ax.cla()
                        self.ax.plot(self.wavelength, self.ydata)
                        #self.ax.plot(self.wavelength, mz)           #MZ
                        plt.draw()
                        plt.pause(0.001)



                        while laser.sweep_status != 0:
                            time.sleep(0.05)  # wait until the laser is ready to go

                        self.stop_flag == 1

                        
                    
                    except KeyboardInterrupt:
                        del readtask
                        break  # The answer was in the question!

                    
                print("...aaand I'm done")

                laser.wavelength = 1550.000  # Set the laser to 1550 nm, can also set it back to the old wl here


            # --------------------------------------------------------------------------
            # Safely close devices
            # --------------------------------------------------------------------------
            finally:
                laser.trigger_output = 'off'

                #readtask.ClearTask()  # clear input channels




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
