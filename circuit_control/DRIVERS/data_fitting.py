import os
import sys
import scipy
import numpy as np
from scipy import optimize
from scipy.optimize import leastsq 
import pandas as pd
import matplotlib.pyplot as plt
import csv
sys.path.append(r"Z:\Lab\YellowLab\Python")


class Fitting_Tools():
        def __init__(self, xdata = None, ydata = None):
                if xdata and ydata is not None:
                        best_parameter_fit = self.fit_data(xdata, ydata)
                        return best_parameter_fit
                else:
                        pass

        @staticmethod
        def linear(x, offset, b):
                return b*x + offset

        @staticmethod
        def quadratic(x, offset, b, c):
                return offset + b*x + c*x**2

        @staticmethod
        def cubic(x, offset, b, c, d):
                return offset + b*x + c*x**2 + d*x**3

        @staticmethod
        def _1gaussian(x, amp1,cen1,sigma1):
                return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen1)**2)/((2*sigma1)**2)))

        @staticmethod
        def _2gaussian(x, amp1,cen1,sigma1, amp2,cen2,sigma2):
                return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen1)**2)/((2*sigma1)**2))) +\
                        amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp(-((x-cen2)**2)/((2*sigma2)**2)))

        @staticmethod
        def _3Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2, amp3,cen3,wid3):
                return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
                        (amp2*wid2**2/((x-cen2)**2+wid2**2)) +\
                        (amp3*wid3**2/((x-cen3)**2+wid3**2))

        @staticmethod
        def single_dip_fit_function_cubic(x, amp1, cen1, wid1, offset, b, c, d):
                return Fitting_Tools._1Lorentz(x, amp1, cen1, wid1) + Fitting_Tools.cubic(x, offset, b, c, d)

        @staticmethod
        def single_dip_fit_function_lin(x,  amp1, cen1, wid1, offset, b):
                return Fitting_Tools._1Lorentz(x, amp1, cen1, wid1) + Fitting_Tools.linear(x, offset, b)

        @staticmethod
        def _1Lorentz(x, amp1, cen1, wid1):
                return (amp1*wid1**2/((x-cen1)**2+wid1**2))
                
        @staticmethod                                           #minimize difference only
        def residuals(p,y,x):
                err = y - Fitting_Tools.single_dip_fit_function_lin(x,*p)
                return err 

        @staticmethod                                   #minimize squared error
        def cost(p, y, x):      
                error = (np.power(Fitting_Tools.single_dip_fit_function_lin(x, *p) - y, 2)) / len(x)
                return error

        @staticmethod
        def lorentzfit(fit_xdata, fit_ydata, p, direction = -1):
                # optimization #
                #pbest = leastsq(Fitting_Tools.residuals, p, args=(fit_ydata,fit_xdata),full_output=1)
                pbest = leastsq(Fitting_Tools.residuals, p, args=(fit_ydata,fit_xdata),full_output=1)
                best_parameters = pbest[0]
                return best_parameters

        @staticmethod
        def residuals_exp(p, y, x):
                err = y - Fitting_Tools.exponential(x, *p)
                return err


        
        @staticmethod
        def exponential(x, a, b):
                return a*(1-np.exp(b*x))
        
        @staticmethod
        def residuals_sigmoid(p, y, x):
                err = y - Fitting_Tools.sigmoid(x, *p)
                return err


        @staticmethod
        def sigmoid(x, amp , scaling,  translation, d):
                return amp*1/(1 + d*np.exp(scaling*(x-translation)))
        
        @staticmethod
        def sigmoid_fit(fit_xdata, fit_ydata, p):
                pbest = leastsq(Fitting_Tools.residuals_sigmoid, p,  args=(fit_ydata,fit_xdata), full_output=1)
                best_parameters = pbest[0]
                return best_parameters

        @staticmethod
        def fit_sigmoid(fit_xdata, fit_ydata, p_0):
            
                pbest = Fitting_Tools.sigmoid_fit(fit_xdata, fit_ydata, p_0)
                return pbest

        @staticmethod
        def exp_fit(fit_xdata, fit_ydata, p):
                pbest = leastsq(Fitting_Tools.residuals_exp, p, args=(fit_ydata,fit_xdata), full_output=1)
                best_parameters = pbest[0]
                return best_parameters

        @staticmethod
        def fit_exp(xdata, ydata):
                p = [-4, 2]
                pbest = Fitting_Tools.exp_fit(xdata, ydata, p)
                return pbest

        @staticmethod
        def fit_data(xdata, ydata, number_of_peaks = 1):
                """calculates a lorentzfit with const background. returned variable is array with
                fit parameters [hwhm, peak center, intensity, offset]. Direction is used to differentiate between 
                dips (-1) and peaks (+1). Note: Find a better fitting algorythm to get rid of this!"""

        
                Amp = max(ydata)-min(ydata)
                fit_center = xdata[np.argmin(ydata)]                        #minimum as peak center
                linewidth = (xdata[-5]+xdata[5])/3 # -1  * np.fabs(np.amax(ydata) - np.amin(ydata))       #intensity of recorded data

                fit_offset = (ydata[5] + ydata[-5]) / 2                     #average in case of non-constant background
                fit_lin_coeff = (ydata[5]-ydata[-5])/(xdata[5]-xdata[-5])
                p = [Amp, fit_center, abs(linewidth), fit_offset, fit_lin_coeff] # [Amp, center, linewidth, offset, linear noise]

                #call fitting lib
                best_parameters = Fitting_Tools.lorentzfit(xdata, ydata, p)

                
                return best_parameters
        

        @staticmethod
        def plot_fit_and_background(x_data, y_data, params):
                pars_lorentz = params[0:3]
                pars_background = params[3:]
                lor_peak_1 = Fitting_Tools._1Lorentz(x_data, *pars_lorentz)       #* in front of vector ensures that values are passed to the function by value, not by ref.
                final_fit = Fitting_Tools.single_dip_fit_function_lin(x_data, *params)
                background_noise = Fitting_Tools.linear(x_data, *pars_background)        
                plt.figure(1)
                plt.plot(x_data, lor_peak_1 + background_noise, "g")
                #plt.fill_between(x_array, lor_peak_1.min(), lor_peak_1, facecolor="green", alpha=0.5)
                
                plt.plot(x_data, background_noise, "y")
                #plt.fill_between(x_data, background_noise.min(), background_noise, facecolor="yellow", alpha=0.5)  

                plt.plot(x_data, final_fit, "r")
                #plt.plot(x_data, lor_peak_1, "g")
                plt.plot(x_data, y_data, "b")
                plt.show()
                

        #Toothsaw
    
        @staticmethod
        def toothsaw_function(x, left_bound, left_slope, mid_bound, right_bound, right_slope, offset, background_lin_coeff):
                if np.all(x <= left_bound):
                        return Fitting_Tools.linear(x, offset, background_lin_coeff)
                if np.all(x > left_bound) and  np.all(x < mid_bound):
                        return left_slope * left_bound + Fitting_Tools.linear(x, offset, background_lin_coeff)
                if np.all(x > mid_bound) and  np.all(x < right_bound):
                        return left_slope * (mid_bound - left_bound) + right_slope * (x-mid_bound) + Fitting_Tools.linear(x, offset, background_lin_coeff)
                if np.all(x >= right_bound):
                        return Fitting_Tools.linear(x, offset, background_lin_coeff)
                        
        @staticmethod  
        def toothsaw_init_fit(xdata, ydata):
                p = [-20, -10, 0, 0.1, 1000, 1, 0.1]
                print(len(p))
                pbest = Fitting_Tools.toothsaw_fit(xdata, ydata, p)
                return pbest

                
        @staticmethod
        def toothsaw_fit(fit_xdata, fit_ydata, p):
                pbest = leastsq(Fitting_Tools.toothsaw_residuals, p, args=(fit_ydata,fit_xdata), full_output=1)
                best_parameters = pbest[0]
                return best_parameters
        
        @staticmethod                                           #minimize difference only
        def toothsaw_residuals(p,y,x):
                err = y - Fitting_Tools.toothsaw_function(x,*p)
                return err 
   

def main():
        df = pd.read_csv(r"Z:\People\KvB\1-02_power_measurements\no_supports\side_coupling\power_measurement_1524,3nm0.txt", delimiter = ',')
        print(df)
        x_data = np.asarray(df[' xdata (MHz)'])
        y_data = np.asarray(df[' ydata (MHz)'])
        print(x_data[0:5])
        print(y_data[0:5])
        coeffs = Fitting_Tools.toothsaw_init_fit(x_data, y_data)
        print(coeffs)

        """
        df = pd.read_csv(r"Z:\People\KvB\24-01_new_fiber_resonances\fine_scan_1522_300micW_some_time_later.txt", delim_whitespace=True)
        x_data = np.asarray(df['Piezo'])
        y_data = np.asarray(df['(V)'])
        #print(x_data.size())
        #print(y_data.size)

        V_voa = np.asarray([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
        P = np.asarray([115.7, 111.8, 99.7, 78.0, 50.45, 24.71, 8.31, 1.67, 0.1657, 0.00989, 0.00655])/115.7
        P_log = 10*np.log(P)
        p_0 = [1, 1, 2, 1]
        coeffs = Fitting_Tools.fit_sigmoid(V_voa, P, p_0)
        print(coeffs)
        final_fit_sigmoid= Fitting_Tools.sigmoid(V_voa, *coeffs)
        #print(Fitting_Tools.sigmoid(2, *coeffs))
        print(final_fit_sigmoid[9])
        plt.figure(2)
        plt.plot(V_voa, 10*np.log10(final_fit_sigmoid), "r")
        plt.plot(V_voa, 10*np.log10(P), "y")
        plt.ylabel("P")
        plt.xlabel("V voa")
     

        #best_params = Fitting_Tools.fit_data(x_data, y_data)
        #Fitting_Tools.plot_fit_and_background(x_data, y_data, best_params)

        #data for second calibration
        V_voa_2 = np.linspace(5, 1.5, 36)
        P_2 = np.asarray([0.226, 0.344, 0.529, 0.822, 1.328, 2.77, 3.56, 5.8, 9.3, 14.64, 22.44, 33.6, 48.8, 69, 95.7, 129.4, 171, 221, 281, 351, 432, 522, 623, 732, 849, 973, 1115, 1239, 1375, 1510, 1650, 1790, 1926, 2050, 2173, 2291])/2700
        p_0 = [1, 1, 2, 1]
        coeffs = Fitting_Tools.fit_sigmoid(V_voa_2, P_2, p_0)
        final_fit_sigmoid_2 = Fitting_Tools.sigmoid(V_voa, *coeffs)
        plt.plot(V_voa, 10*np.log10(final_fit_sigmoid_2), "b")
        plt.plot(V_voa_2, 10*np.log10(P_2))
        plt.legend(["fit_1", "data_1", "fit_2", "data_2"])
        plt.show()
        """

if __name__ == '__main__': 
        main()
    

"""
amp1 = -0.3
cen1 = -0.2
sigma1=0.2
offset = 1.0
b = 0.1
c = 0.1
d = 0.1
def 
#fit cubic noise                  
popt_lorentz, pcov_lorentz = scipy.optimize.curve_fit(single_dip_fit_function, x_data, y_data, p0=[amp1, cen1, sigma1, offset, b,c,d])
perr_lorentz = np.sqrt(np.diag(pcov_lorentz))
pars_lorentz = popt_lorentz[0:3]
pars_background = popt_lorentz[3:]
lor_peak_1 = _1Lorentz(x_data, *pars_lorentz)
final_fit = single_dip_fit_function(x_data, *popt_lorentz)
background_noise = cubic(x_data, *pars_background)
print(popt_lorentz)

plt.figure(1)
plt.plot(x_data, lor_peak_1 + background_noise, "g")
#plt.fill_between(x_array, lor_peak_1.min(), lor_peak_1, facecolor="green", alpha=0.5)
  
plt.plot(x_data, background_noise, "y")
#plt.fill_between(x_data, background_noise.min(), background_noise, facecolor="yellow", alpha=0.5)  

plt.plot(x_data, final_fit, "r")
#plt.plot(x_data, lor_peak_1, "g")
plt.plot(x_data, y_data, "b")
#plt.show()

plt.figure(2)
popt_lorentz_lin_fit, pcov_lorentz_lin_fit = scipy.optimize.curve_fit(single_dip_fit_function_lin, x_data, y_data, p0=[amp1, cen1, sigma1, offset, b])
perr_lorentz = np.sqrt(np.diag(pcov_lorentz))
lor_lin_fit = single_dip_fit_function_lin(x_data, *popt_lorentz_lin_fit)

plt.plot(x_data, y_data, "b")
plt.plot(x_data,lor_lin_fit, "r")
plt.show()

"""
"""

popt_2gauss, pcov_2gauss = scipy.optimize.curve_fit(_2gaussian, x_array, y_array_2gauss, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2])
perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
pars_1 = popt_2gauss[0:3]
pars_2 = popt_2gauss[3:6]
gauss_peak_1 = _1gaussian(x_array, *pars_1)
gauss_peak_2 = _1gaussian(x_array, *pars_2)

ax1.plot(x_array, gauss_peak_1, "g")
ax1.fill_between(x_array, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)
  
ax1.plot(x_array, gauss_peak_2, "y")
ax1.fill_between(x_array, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)  
"""


"""


"""
