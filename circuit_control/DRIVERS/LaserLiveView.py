
"""Write smth here this time!
"""
import time
import sys
import os

from PyQt5 import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
 
sys.path.append("Z:\Lab\Pulsing_Setup\python")

import nidaqmx



class LaserLiveView(QtGui.QDialog):
    def __init__(self, parent=None):
        super(LaserLiveView, self).__init__(parent)

        self.layout = QtGui.QGridLayout()
        self.setLayout(self.layout)     
        
        pg.setConfigOption('background', 0.3)
        pg.setConfigOption('foreground', 'w')
        
        self.plot1 = pg.PlotWidget()
        self.layout.addWidget(self.plot1, 0, 0, 1, 4)
        
        self.btn_quit = QtGui.QPushButton('Quit')
        self.layout.addWidget(self.btn_quit, 1, 3, QtCore.Qt.AlignRight)   # button goes in upper-left
        self.btn_quit.clicked.connect(self.savequit)
        
        self.btn_reset = QtGui.QPushButton('Reset')
        self.layout.addWidget(self.btn_reset, 2, 0, QtCore.Qt.AlignLeft)   # button goes in upper-left
        self.btn_reset.clicked.connect(self.resetplot)
  
        self.comboBox = QtGui.QComboBox(self)
        self.layout.addWidget(self.comboBox, 2, 2, QtCore.Qt.AlignLeft)
        self.comboBox.addItems(["ai" + str(i) for i in range(7)])
        self.comboBox.activated.connect(self.switch_channel)
    
        self.btn_startstop = QtGui.QPushButton('Start / Stop')
        self.layout.addWidget(self.btn_startstop, 1, 0, QtCore.Qt.AlignLeft)   # button goes in upper-left
        
        self.btn_startstop.clicked.connect(self.startstop)
        

        
        self.curve = self.plot1.plot(pen='y')

        self.len_list = 0  
        self.wlm_op_state = 0
        self.start_time = time.time()
        
        self.xdata = []
        self.ydata = []
    
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)

        

    def switch_channel(self):
        text = self.comboBox.currentText()
        self.channel = "cDAQ1Mod3/" + text
        try:    
            self.readtask.close()           
        except:
            pass
        self.resetplot()
        self.readtask = nidaqmx.task.Task()
  
        self.readtask.ai_channels.add_ai_voltage_chan(self.channel, min_val=-.5, max_val=10)
        print('Now reading ' + self.channel)

    def update(self):
        
        if (self.wlm_op_state == 1):
            
            voltage = self.readtask.read()
            #plotting
            self.xdata.append(time.time()-self.start_time)
            self.ydata.append(voltage)
            
            
            if self.len_list > 500:
                del self.xdata[0]
                del self.ydata[0]
            else:
                self.len_list += 1
                
            self.curve.setData(self.xdata, self.ydata)
            
    def resetplot(self):
        self.len_list = 0
        self.xdata = []
        self.ydata = []
        
    def startstop(self): 
   
        if (self.wlm_op_state == 0):
            try:
                self.channel
            except AttributeError:
                self.switch_channel()

            self.len_list = 0
            self.xdata = []
            self.ydata = []
            self.start_time = time.time()
            self.wlm_op_state = 1
            self.timer.start(100)

        else:      
            self.wlm_op_state = 0
            self.timer.stop()
            self.readtask.close()
            

    def savequit(self):
        print("savequit")
        self.timer.stop()
        self.readtask.stop()
        print("savequit2")
        exit()
        print("savequit2")

if __name__ == '__main__': 
    
    app = QtGui.QApplication([])
    
    Window = LaserLiveView()       
    #Window.setQuitOnLastWindowClosed(True)    
    Window.show()

    def checkd():
        Window.save_exit()
        print('exit here')
    app.lastWindowClosed.connect(checkd)
    app.exec_()
