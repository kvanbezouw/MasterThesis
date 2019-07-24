import sys
sys.path.append(r"Z:\Lab\YellowLab\Python")
from DRIVERS.yellow_cDAQ import cDAQtest

def calib_output_voltage(read_channel = "ai0", write_channel = "ao0", v_set_to = 2, accuracy = 0.007, calibrated = None):
    channel_in = "cDAQ1Mod3/" + read_channel
    channel_out_voa = "cDAQ1Mod1/" + write_channel
    v_set_to = float(v_set_to)
    accuracy =float(accuracy)
    error_list = []
    v_voa = float(5)
    cDAQtest.cDAQ_write_DC(channel = channel_out_voa, voltage = v_voa)                  #set voa to 5 by default to prevent photodiode from blowing up main
    try:
        assert v_set_to <= 3                          #amplifier can't handle a voltage more than 3V
        assert v_set_to >= 0
        
        #ideally sense the voa voltage, then apply voltage if needed. Work on this later. Now make sure to set VOA at 5V initially. Doesn't really matter though.
        counter = 0
        while calibrated == None:         #This is a terrible way to make the while loop but I'm tired so leave it at this for now
            v_actual= cDAQtest.cDAQ_read(channel = channel_in) 

            #print("v_voa = ", str(v_voa))
            cDAQtest.cDAQ_write_DC(channel = channel_out_voa, voltage = v_voa)
            error_list.append(abs(v_actual- v_set_to))
           # print(error_list[counter])
            if error_list[counter] < accuracy:
                calibrated = []
                print("Photodiode voltage calibrated to {} V by a VOA voltage of {} within a range of {} V in {} steps.".format(str(v_set_to), str(v_voa), str(2*accuracy), str(counter)))
                break
            if v_actual > v_set_to:
                v_voa = v_voa + 0.008
            if v_actual < v_set_to:
                v_voa = v_voa - 0.008
           
            counter = counter + 1

            if counter == 150:
                if (error_list[0] - error_list[counter-2]) <= 0.010:
                    print("Calibration is not progressing or is stuck. Probably selected wrong ports, or signal is too small so output is independent of VOA anyway.")
                    print("VOA output set to 5V")
                    cDAQtest.cDAQ_write_DC(channel = channel_out_voa, voltage = 5) #Just to make sure that PDs are not getting overloaded
                    break
        
    except AssertionError:
        print("Desired photodiode voltage out of range")
        pass

         
                

        
