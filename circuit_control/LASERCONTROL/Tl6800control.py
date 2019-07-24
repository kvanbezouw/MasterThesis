import ctypes
import os
import numpy as np
import struct
import pickle
import time
import datetime
import multiprocessing
import math


class TL6800():

    def __init__(self,P_ID = "100A", DevID = 1, wavelength_start = 1521, wavelength_stop = 1570, fwdvel = 10, bwdvel = 10, scancfg = 1):
        self.TLDLL = ctypes.CDLL("UsbDll.dll")
        self.TL6800_initialise_defaultvalues(P_ID, DevID, wavelength_start, wavelength_stop, fwdvel, bwdvel, scancfg)
        self.TL6800_startup_device()
        #self.TL6800_sweep_parameters(printbool = False)
    
    def TL6800_initialise_defaultvalues(self, P_ID, DevID, wavelength_start, wavelength_stop, fwdvel, bwdvel, scancfg):
        self.P_ID = P_ID 
        self.DevID = DevID
        self.wavelength_start = wavelength_start
        self.wavelength_stop = wavelength_stop
        self.fwdvel = fwdvel
        self.bwdvel = bwdvel
        self.scancfg = scancfg

    def TL6800_startup_device(self):
        try:
            self.TL6800_read_ascii()
        except OSError:
            print("TL6700 reading error caught and passed")
            pass
        self.TL6800_OpenAllDevices() 
        self.TL6800_OpenDevice()
    
        #Sometimes the buffer needs to get rid off some output data, sometimes it doesn't, in which case it gives an OSError
        try:
            self.TL6800_read_ascii(printbool = False)
        except OSError:
            print("TL6700 reading error caught and passed")
            pass

    def TL6800_sweep_parameters(self, printbool = True):
        #SOURce:WAVE:START xxxx, SOURce:WAVE:STOP xxxx, SOURce:WAVE:SLEW:FORWard xx, SOURce:WAVE:SLEW:RETurn xx
        self.TL6800_sweep_wavelength_start(printbool = printbool)
        self.TL6800_sweep_wavelength_stop(printbool = printbool)
        self.TL6800_sweep_set_forwardvelocity(printbool = printbool)
        self.TL6800_sweep_set_backwardvelocity(printbool = printbool)
        
    def TL6800_sweep_start(self, printbool = True, readbool = True): #mind to put trackingmode on, prior
        self.TL6800_set_trackmode(printbool = printbool)
        command = 'OUTPut:SCAN:START'
        self.TL6800_write_ascii(command, printbool = printbool, readbool = readbool)

        
    def TL6800_sweep_wavelength_start(self, printbool = True):
        strwavelength_start = str(self.wavelength_start)
        command = 'SOURce:WAVE:START ' + strwavelength_start
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_sweep_wavelength_stop(self, printbool = True):
        strwavelength_stop = str(self.wavelength_stop)
        command = 'SOURce:WAVE:STOP ' + strwavelength_stop
        self.TL6800_write_ascii(command, printbool = printbool)


    def TL6800_sweep_set_forwardvelocity(self, fwdvel = None, printbool = True):         # [fdwvel] = nm/s
        if(fwdvel == None):  
            str_fwdvel = str(self.fwdvel)
            #print(str_fwdvel)
        else:
            str_fwdvel = str(fwdvel)
            print(str_fwdvel)
        command = 'SOURce:WAVE:SLEW:FORWard ' + str_fwdvel
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_sweep_set_backwardvelocity(self, bckwdvel = None, printbool = True):         # [fdwvel] = nm/s
        if(bckwdvel == None):  
            str_bwdvel = str(self.bwdvel)
        else:
            str_bwdvel = str(bckwdvel)
        command = 'SOURce:WAVE:SLEW:RETurn ' + str_bwdvel
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_set_piezo_voltage(self, piezovoltage, printbool = True):
        piezovolt = str(piezovoltage)
        command = 'SOURce:VOLTage:PIEZo ' + piezovolt
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_query_piezo_voltage(self, printbool = True):
        command = 'SOURce:VOLTage:PIEZo?'
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_set_brightness(self,  brightness):
        strbrightness = str(brightness)
        command ='BRIGHT ' + strbrightness               
        self.TL6800_write_ascii(command)

    def TL6800_query_wavelength(self, printbool = True):
        command = "SENSe:WAVElength"
        RawLambda = self.TL6800_write_ascii(command, printbool = printbool)
        #Returns lambda as "'xxxx.xxx\r\n..". Only keep the wavelength (xxxx.xxx)
        LambdaDigit = RawLambda[1:9]    
        return LambdaDigit

    def TL6800_query_trackmode(self, printbool = False):
        command = "OUTPut:TRACk?"
        readbuffer = self.TL6800_write_ascii(command, printbool = printbool)
        trackmode = int(readbuffer[1])
        print("trackmode is: ", trackmode)
        try: 
            if trackmode == 1:
                trackbool = True
                return trackbool
            if trackmode == 0:
                trackbool = False
                return trackbool
        except:
            pass
            
    def TL6800_query_power(self):
        command = 'SENSe:POWer:DIODe'
        RawPower = self.TL6800_write_ascii(command)
        PowerDigit = RawPower[1:5]
        return PowerDigit

    def TL6800_query_current(self):
        command = 'SENSe:CURRent:DIODe'
        RawCurrent = self.TL6800_write_ascii(command)
        LaserDigit = RawCurrent[1:5]
        return LaserDigit

    def TL6800_set_wavelength(self, wavelength, printbool = True):
        wavelength = str(wavelength)
        command = 'SOURce:WAVElength ' + wavelength
        #toggle track mode on, otherwise actual output doesnt change
        self.TL6800_set_trackmode(onness = 1, printbool = printbool)
        self.TL6800_write_ascii(command, printbool = printbool)

    def TL6800_set_powermode(self, poweronness = 1):
        poweronness = str(poweronness)
        if poweronness == 1:
            command = 'SOURce:CPOWer ON'
        if poweronness == 0:
            command = 'SOURce:CPOWer OFF'
        self.TL6800_write_ascii(command)



    def TL6800_set_trackmode(self, onness = 1, printbool = True):                #Toggles Wavelength Track Mode (allows you to vary lambda).
    
        if onness == 1:
            command = 'OUTPut:TRACk ON'
        if onness == 0:
            command = 'OUTPut:TRACk OFF' 
            
        self.TL6800_write_ascii(command, printbool = printbool)

        if printbool:
            if onness == 1:
                print("Tracking mode has been turned ON")
            if onness == 0:
                print("Tracking mode has been turned OFF")
        
       
    def TL6800_set_power(self, power):  # [mW]
        command = 'SOURce:POWer:DIODe ' + str(power)    
        self.TL6800_write_ascii(command)

    def TL6800_set_current(self, current): #mA
        command = 'SOURce:CURRent:DIODe ' + str(current)
        self.TL6800_write_ascii(command)

     
    def TL6800_write_ascii(self, command, printbool = True, readbool = True):
  
        long_deviceid = ctypes.c_long(self.DevID)
        char_command = ctypes.create_string_buffer(command.encode())
        
        length = ctypes.c_ulong(len(char_command))
        self.TLDLL.newp_usb_send_ascii(long_deviceid,char_command,length)               #newp_usb_send_ascii (long DeviceID, char* Command, unsigned long Length);
       
        if printbool:
            print("Command sent to mr. TL6800: ",char_command.raw)

        #Reading the output that the laser returns is necessary not to get reading errors in second trials. For timing processes, it may not be preferred (mind to read after)
        if readbool:
            output = self.TL6800_read_ascii(printbool = printbool)
            return output
        

    def TL6800_read_ascii(self, printbool = True):
        long_deviceid = ctypes.c_long(self.DevID)
        Buffer = ctypes.create_string_buffer(1024)                                        #ctypes.create_string_buffer(b"*IDN?",1024)
        Length = ctypes.c_ulong(len(Buffer))
        #BytesRead = 1024
        BytesRead = ctypes.create_string_buffer(1) 
        self.TLDLL.newp_usb_get_ascii(long_deviceid, Buffer, Length, BytesRead) # newp_usb_get_ascii (long DeviceID, char* Buffer, unsigned long Length, unsigned long* BytesRead);
        if printbool:
            print("Response sent by device: ",repr(Buffer.raw).split("\\x")[0][1:])
        output = repr(Buffer.raw).split("\\x")[0][1:]
        return output


    def TL6800_device_info(self):
        szDevInfo = ctypes.create_string_buffer(1024)
        self.TLDLL.newp_usb_get_device_info(szDevInfo)     # newp_usb_get_device_info (char* Buffer);
        #print(repr(szDevInfo.raw))
        #pretty elaborate, but this prints the device info only
        print("Made connection to: ", repr(szDevInfo.raw).split("\\x")[0][1:][3:(len(repr(szDevInfo.raw).split("\\x")[0][1:])-1)]) 
        return szDevInfo
        
    def TL6800_OpenAllDevices(self):
        self.TLDLL.newp_usb_init_system()

    def TL6800_OpenDevice(self):
        self.TLDLL.newp_usb_init_product(self.P_ID)

    def TL6800_CloseDevices(self):
        self.TLDLL.newp_usb_uninit_system()

    def TL6800_scancfg(self):         # 0 = do not reduce laser output to 0 during reverse scan, 1 = reduce laser output to 0 drs
        command = 'SOURce:WAVE:SCANCFG ' + str(self.scancfg)    
        self.TL6800_write_ascii(command)
     
    def TL6800_beep(self):
        command = 'BEEP'
        #printbool false because beep does not return anything
        self.TL6800_write_ascii(command, printbool = False)

    
         
def main():
    tl6800 = TL6800()
    #tl6800.TL6800_set_wavelength(wavelength = 1535.8)

if __name__ == '__main__': 
    main()
    
