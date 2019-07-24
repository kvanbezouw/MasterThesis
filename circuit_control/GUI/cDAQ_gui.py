from PyQt5 import QtWidgets

import sys

sys.path.append(r"Z:\Lab\YellowLab\Python")
import nidaqmx
from DRIVERS.yellow_cDAQ import cDAQtest           #Contains tasks to send/read from

from PyQt5 import QtCore, QtGui, QtWidgets
from cDAQ_gui_design import Ui_MainWindow
#Update UI file via the command; pyuic5 cDAQgui.ui > xyz.py
import time

 
cDAQtest = cDAQtest()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        self.setupUi(self)
       
        #This creates actions resulting in a change in voltage in module 1 (output). Can be copypastad for changes in voltages in module 2(output)
        self.cDAQ1Mod1ao0OLabel.returnPressed.connect(lambda: self.on_enter(cDAQtest, "ao0"))
        self.cDAQ1Mod1ao1OLabel.returnPressed.connect(lambda: self.on_enter(cDAQtest, "ao1"))
        self.cDAQ1Mod1ao2OLabel.returnPressed.connect(lambda: self.on_enter(cDAQtest, "ao2"))
        self.cDAQ1Mod1ao3OLabel.returnPressed.connect(lambda: self.on_enter(cDAQtest, "ao3"))

        self.checkBoxcDAQ1Mod3LiveDat.stateChanged.connect(self.live_feed)

        self.actionMANN.triggered.connect(self.about_message) 
        
    def about_message(self, hopen = 4):
        QtWidgets.QMessageBox.about(self,"About","Store information about how to handle the program here, eventually.")
          
    
    def live_feed(self, state):
        if state == QtCore.Qt.Checked: 
                readvalue0 = str(cDAQtest.cDAQ_read("cDAQ1Mod3/ai0"))
                readvalue0 = readvalue0[0:8]
                self.cDAQ1Mod3ai0InputLabel.setText(readvalue0)

                readvalue1 = str(cDAQtest.cDAQ_read("cDAQ1Mod3/ai1"))
                readvalue1 = readvalue1[0:8]
                self.cDAQ1Mod3ai1InputLabel.setText(readvalue1)

                readvalue2 = str(cDAQtest.cDAQ_read("cDAQ1Mod3/ai2"))
                readvalue2 = readvalue2[0:8]
                self.cDAQ1Mod3ai2InputLabel.setText(readvalue2)

                readvalue3 = str(cDAQtest.cDAQ_read("cDAQ1Mod3/ai3"))
                readvalue3 = readvalue3[0:8]
                self.cDAQ1Mod3ai3InputLabel.setText(readvalue3)
                
        else:
            pass

    def on_enter(self,cDAQ, channel):
        #voltage = self.cDAQ1Mod1ao0OLabel.text()
        channel = str(channel)
        if channel == "ao0":
            voltage = str(self.cDAQ1Mod1ao0OLabel.text())                #this may lead to genericness
            print(voltage)
            self.cDAQ1Mod1ao0COLabel.setText(voltage)
            cDAQ.cDAQ_write_DC("cDAQ1Mod1/ao0", voltage = voltage)
            self.cDAQ1Mod1ao0COLabel.setText(voltage)
        if channel == "ao1":
            voltage = str(self.cDAQ1Mod1ao1OLabel.text())                #this may lead to genericness
            print(voltage)
            self.cDAQ1Mod1ao1COLabel.setText(voltage)
            cDAQ.cDAQ_write_DC("cDAQ1Mod1/ao1", voltage = voltage)
            self.cDAQ1Mod1ao1COLabel.setText(voltage)
        if channel == "ao2":
            voltage = str(self.cDAQ1Mod1ao2OLabel.text())                #this may lead to genericness
            print(voltage)
            self.cDAQ1Mod1ao2COLabel.setText(voltage)
            cDAQ.cDAQ_write_DC("cDAQ1Mod1/ao2", voltage = voltage)
            self.cDAQ1Mod1ao2COLabel.setText(voltage)
        if channel == "ao3":
            voltage = str(self.cDAQ1Mod1ao3OLabel.text())                #this may lead to genericness
            print(voltage)
            self.cDAQ1Mod1ao3COLabel.setText(voltage)
            cDAQ.cDAQ_write_DC("cDAQ1Mod1/ao3", voltage = voltage)
            self.cDAQ1Mod1ao3COLabel.setText(voltage)
            
   

def main():
    app = QtWidgets.QApplication(sys.argv)  # A new instance of QApplication
    form = MainWindow()                 # We set the form to be our ExampleApp (design)
    form.show()                         # Show the form
    app.exec_()                         # and execute the app


if __name__ == '__main__':              # if we're running file directly and not importing it
    main()                              # run the main function
