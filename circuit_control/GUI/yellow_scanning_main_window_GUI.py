# -*- coding: utf-8 -*-
"""
Spyder Editor

This is the main GUI builder file. Run it in order to start the test software.
"""

import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
from DRIVERS.CalibrateOutputVoltage import calib_output_voltage
from LASERCONTROL.Tl6800control import TL6800
import os.path
import subprocess
from PyQt5 import QtGui, QtCore
import numpy as np
from DRIVERS.yellow_cDAQ import cDAQtest
from os.path import abspath, dirname, join
from GUI.yellow_fine_scan_gui import fine_scan_GUI
from GUI.yellow_coarse_scan_gui import coarse_scan_GUI
from DRIVERS.LaserLiveView import LaserLiveView
import GUI.cDAQ_gui_design
import pyqtgraph as pg
import time

class Window(QtGui.QMainWindow, cDAQtest):
    """Builds the Qt GUI by initializing the QMainWindow."""
    def __init__(self):
        super(Window, self).__init__()
        mainMenu = self.menuBar()
        #Task Bar Action initialization

        plotTracesAction = QtGui.QAction('&View Scans', self)
        plotTracesAction.triggered.connect(self.plot_traces)

        switchesAction = QtGui.QAction('&Switch control', self)
        switchesAction.triggered.connect(self.switch_switches)
        
        vattenAction = QtGui.QAction('&VOA control', self)
        vattenAction.triggered.connect(self.vatten_control)        

        closeAction = QtGui.QAction('&Secure Quit', self)
        closeAction.triggered.connect(self.close_application)
        
        getLiveViewAction = QtGui.QAction('Live View', self)
        getLiveViewAction.triggered.connect(self.get_live_view)

        
        
        setLaserPowerAction = QtGui.QAction('Set Laser Power', self)
        setLaserPowerAction.triggered.connect(self.set_laser_power)
        
        setLaserWavelength = QtGui.QAction('Set Laser Wavelength', self)
        setLaserWavelength.triggered.connect(self.set_laser_wavelength)

        setLaserCurrentAction = QtGui.QAction('Set Laser Current', self)
        setLaserCurrentAction.triggered.connect(self.set_laser_current)

        self.toggleSwitchAction = QtGui.QAction('Toggle Switch', mainMenu, checkable=True)
        self.toggleSwitchAction.triggered.connect(self.toggle_switch)

        setSwitchAoChannel = QtGui.QAction('Set switch output port', self)
        setSwitchAoChannel.triggered.connect(self.set_switch_ao_channel)

        calibrateVOAs = QtGui.QAction('Calibrate photodiode output', self)
        calibrateVOAs.triggered.connect(self.calibrate_pd_via_voa)
        
        #Task Bar Interface initialization
      
        fileMenu = mainMenu.addMenu('&File')
        fileMenu.addAction(plotTracesAction)
        fileMenu.addAction(switchesAction)
        fileMenu.addAction(vattenAction)

        fileMenu.addAction(closeAction)

        setMenu = mainMenu.addMenu('&Laser')
        setMenu.addAction(getLiveViewAction)
        setMenu.addAction(setLaserPowerAction)
        setMenu.addAction(setLaserWavelength)
        setMenu.addAction(setLaserCurrentAction)

        setMenu = mainMenu.addMenu('&VOAs')
        setMenu.addAction(calibrateVOAs)
        
     
        setMenu = mainMenu.addMenu('&Switch')
        setMenu.addAction(self.toggleSwitchAction)
        setMenu.addAction(setSwitchAoChannel)

        self.switch_channel = "cDAQ1Mod1/ao1"
        self.home()
        self.show()
        
        
    def home(self):
        """Build the main interface here"""
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
        pg.ValueLabel(siPrefix = True, formatStr = None)
        # main layout
        self.mainLayout = QtGui.QVBoxLayout()

        # add all main to the main vLayout
        self.mainLayout.addWidget(coarse_scan_GUI())
        self.mainLayout.addWidget(fine_scan_GUI())
        #self.mainLayout.addWidget(self.scrollArea)

        # central widget
        self.centralWidget = QtGui.QWidget()
        self.centralWidget.setLayout(self.mainLayout)

        # set central widget
        self.setCentralWidget(self.centralWidget)
        
        
    ###########################################################################
    #Task Bar helper functions            
    def toggle_switch(self): #5V is transmit, 0V is reflect
        if self.toggleSwitchAction.isChecked():
            print("switch turned on")
            self.cDAQ_write_DC(channel = self.switch_channel, voltage = 5)
        else:
            print("switch turned off")
            self.cDAQ_write_DC(channel = self.switch_channel, voltage = 0)

    def set_switch_ao_channel(self):
        channels = ["ao1"]  #This is the default (connected on 20/12/18)
        channels.append("ao0")
        for i in range(2,9):
            channels.append("ao" + str(i))
        
        item, ok = QtGui.QInputDialog.getItem(self, "Switch control", "Select an output channel", channels, 0, False)

        if ok and item:
            self.switch_channel = "cDAQ1Mod1/" + str(item)
            print("switch set to: ", self.switch_channel)


    def get_live_view(self):    
        """Calls helper function to show live measurement of the PD"""
        liveViewWindow = LaserLiveView()
        liveViewWindow.exec_()

    def plot_traces(self):
        subprocess.Popen(r'python Z:\Lab\Pulsing_Setup\python\tools\trace_db_2_0.py', shell=True)
        #os.system("plot_traces.py 1")

    def switch_switches(self):
        #os.chdir("\\groeblachernas.synology.me\higgs\Lab\Pulsing_Setup\python\tools")
        subprocess.Popen(r'python Z:\Lab\Pulsing_Setup\python\tools\switches_3_1.py', shell=True)

    def vatten_control(self):
        #os.chdir("\\\groeblachernas.synology.me\higgs\Lab\Pulsing_Setup\python\laser_control")
        #sys.path.append("Z:\Lab\Pulsing_Setup\python\laser_control")
        #subprocess.Popen("voas_1_0.py 1", stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd="Z:/Lab/Pulsing_Setup/python",shell=True)
        #path = abspath(join(dirname(__file__), '../subdir1/some_executable'))
        #os.chdir("Z:\Lab\Pulsing_Setup\python\laser_control")
        #print(os.getcwd())
        #subprocess.Popen("python voas_1_0.py 1", cwd = "Z:\Lab\Pulsing_Setup\python\tools", shell=True)
        subprocess.Popen("GUI\cDAQ_gui.py", shell = True)

        #in cDAQGUI, executing main shows the GUI

    def set_laser_power(self):
        """Set the power of the laser"""
        try:
            old_power_value =  tl6800.TL6800_query_power() #Read laser power
        except:
            #Proper error catching to be done                        
            pass  
        value, ok = QtGui.QInputDialog.getText(self, 'Input Dialog', 
            'Enter laser power (mW):', QtGui.QLineEdit.Normal, str(old_power_value))
        if ok:
            try:
                value = float(value)
                value = round(value, 3) #Rounding prob not necessary
                if (value > 0) and (value < 15): #Limits for laser power, dont change
                    try:
                        tl6800.TL6800_set_power(power = value)
                        print("Lasahr power set to: ", str(value))                    
                    except:
                        #Proper error catching to be done                        
                        pass    
                else:
                    #Proper error catching to be done                        
                        print('wrong scale')
            except ValueError:
                #Proper error catching to be done                        
                print('Wrong input. Try a power between 0 and 15 [mW]')

    def set_laser_current(self):
        try:
            old_current_value =  tl6800.TL6800_query_current() #Read laser power
        except:
            #Proper error catching to be done                        
            pass  
        value, ok = QtGui.QInputDialog.getText(self, 'Input Dialog', 
            'Enter laser current (mA):', QtGui.QLineEdit.Normal, str(old_current_value))
        if ok:
            try:
                value = float(value)
                if (value > 0) and (value < 200): #Limits for laser power, dont change
                    try:
                        tl6800.TL6800_set_current(current = value)
                        print("Lasahr current set to: ", str(value))                    
                    except:
                        #Proper error catching to be done                        
                        pass    
                else:
                    #Proper error catching to be done                        
                    print('wrong scale')
            except ValueError:
                #Proper error catching to be done                        
                print('Wrong input. Try a current between 0 and 200 [mA]')


                
    def set_laser_wavelength(self):
        """Set the wavelength of the laser"""

        old_value = tl6800.TL6800_query_wavelength() #Read laser power
        print(old_value)

        value, ok = QtGui.QInputDialog.getText(self, 'Input Dialog', 
            'Enter wavelength (nm):', QtGui.QLineEdit.Normal, str(old_value))
        if ok:
            try:
                value = float(value)
                #Laser range for tl6800
                if (value > 1520) and (value < 1570):
                    print("Via GUI, the wavelength is set to", str(value))
                    try:
                        tl6800.TL6800_set_wavelength(wavelength = value)                       
                    except Exception as e:
                        print("Something went wrong when setting the wavelength. You caught the following error:\n", e)                        
                        pass
                else:
                    #Proper error catching to be done                        
                    print('wrong scale')
            except ValueError:
                #Proper error catching to be done                        
                print('wrong input')
            pass
            
    def calibrate_pd_via_voa(self):
        #ask for ai, ao and go calibrate mi mangg
        channels_ai = []
        channels_ao = []
        for i in range(0,9):
            channels_ao.append("ao" + str(i))
            channels_ai.append("ai" + str(i))

        set_voltage_to, ok_1 = QtGui.QInputDialog.getText(self, "VOA Control", 'Set output voltage to what value [V]?', QtGui.QLineEdit.Normal)
        if ok_1:
            picked_ao_channel, ok_2 = QtGui.QInputDialog.getItem(self, "VOA control", "Select the VOA ao port", channels_ao, 0, False)
            print(picked_ao_channel)
            if ok_2:
                picked_ai_channel, ok_3 = QtGui.QInputDialog.getItem(self, "VOA control", "Select the output ai port", channels_ai, 0, False)
                print(picked_ai_channel)
                if ok_3:
                    print("use channel {} to calibrate channel {}".format(str(picked_ao_channel),str(picked_ai_channel)))
                    calib_output_voltage(read_channel = picked_ai_channel, write_channel = picked_ao_channel, v_set_to = set_voltage_to)
                    print("VOA calibrated")
                


    def close_application(self):
        """Function to savely close the software. Most importantly close the wavelengthmeter. 
        Check if it is possible to call this with the regular 'x' button."""

        wlm = Wavelengthmeter()
        wlm.CloseWLM()

        sys.exit()
    

    
def run():
    """Runs the App"""
   
    app = QtGui.QApplication([])
    app.setStyle("plastique")
    GUI = Window()
    GUI.setWindowTitle("Laser Scan")
    sys.exit(app.exec_())



#initialise laser
tl6800 = TL6800()
run()
