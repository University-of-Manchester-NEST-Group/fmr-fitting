# -*- coding: utf-8 -*-
"""
A collection of methods to process and fit Ferromagnetic Resonance Data 
(as collected from the Vector Network Analyzer - Ferromagnetic Resonance setup at University of Manchester)
Created on Mon Jun 22 14:17:21 2020
@author: Harry Waring
"""
## Call needed packages
#general
import numpy as np
import math
import csv

#Fit
import lmfit
from lmfit import Minimizer, Parameters, report_fit, Model, Parameter
import scipy as scipy
from scipy.interpolate import interp1d
from scipy import signal, array, polyfit
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec
import matplotlib.ticker as ticker
from scipy.stats import linregress
#import urllib.request




''' A collection of functions to analyse s12 data collected from VNA-FMR experiments.
Can correct backgound and fit up to four resonant peaks'''
    
#Background removal (Relies on a region outside resonance range to take as reference measurement)
#For more sensitive measurments, the user is reccommended to use methods whoch remove resonant peaks from the data set
def _BaselineRemoval(s12C, excludeRange):
    
    #Find index of minumim values in frequency vs s12 (to find baseline)
    i=0
    minValuesIdx=[]
    while i < len(s12C[:,1]):
        minValuesIdx.append(np.argmin(s12C[i,:]))
        i+=1
    
    baselines= []
    d=0
    #Average s12 data up to mimumin frequency (neccessary to remove resonance region)
    while d<np.shape(s12C)[0]:
            meanValue= np.mean(s12C[d,0:(minValuesIdx[d]-excludeRange)])
            baselines.append(meanValue)
            d+=1


    correctedS12 = np.zeros(np.shape(s12C)[1])
    
    #Remove baseline from s12 data
    z=0
    while z<(np.shape(s12C)[0]):
        correctedS12 = np.vstack((correctedS12, s12C[z,:] - baselines[z]))
        z+=1

    correctedS12 = np.delete(correctedS12, (0), axis=0)   
    return correctedS12




''' Field: Calculate applied external static field from calibration to current in electromagnet '''
''' NOTE: 04/07/2022 limits of fieldCurrentFile adjusted to match size of current file'''
''' NOTE: 05/07/2022 added fill_value=extrapolate to fix unknown error with x_new limits'''
def _calcField(fieldCurrentFile, current):
    f = interp1d(fieldCurrentFile[0:250,0],fieldCurrentFile[0:250,1],kind = 'linear', fill_value='extrapolate')
    field = f(current[:,0])
    return field


''' Lorentz (Real and imaginary) Equation: Define equation used for lorentzian'''
# x=frequency, y0=baseline, amp1=peak amplitude, cen1= Resonant Frequency, c1 = mixing parameter
def _1Lorentz( x, y0,  amp1, wid1, cen1, c1):
    return y0 + (amp1*wid1**2/(4*(x-cen1)**2+wid1**2)) + ((c1*amp1*wid1**2*(x-cen1))/((((x-cen1)**2 + wid1**2)**2)))

''' Derived Fitting Parameter: Area '''
    # If it is not possible to calaulate set to None.
def _1calcPeakArea( amp,wid, amp_err, wid_err, mixing, mixing_err):
    
    if amp is not None and wid is not None:
        area= amp*math.pi*wid/2
        
    else:
        area=""
        #area_err = ""
        
    if  amp_err is not None and wid_err is not None:
        area_err = (area)*np.sqrt(((amp_err)/(amp))**2 + ((wid_err)/(wid)**2))
    else: 
        area_err = ""
    return area, area_err


'''Lorentz Fitting Model: Allows up to four resonances (real and imaginary components) to be fitted'''
# Phi = mixing (See other parameter meanings above) 
# Recieves data and expected parameters
# Retruns residual
def LorentzFit( params, x, data):
    v = params.valuesdict()
    model = (v['y0'] + (v['m']*x)) + np.cos(v['phi_1'])*v['amp1']*(v['wid1']**2/((x-v['cen1'])**2+v['wid1']**2)) + \
            ((np.sin(v['phi_1'])*v['amp1']*v['wid1']*(x-v['cen1']))/((((x-v['cen1'])**2 + v['wid1']**2)))) + \
              (np.cos(v['phi_2'])**2)*v['amp2']*(v['wid2']**2/((x-v['cen2'])**2+v['wid2']**2)) + \
            ((np.sin(v['phi_2'])*v['amp2']*v['wid2']*(x-v['cen2']))/((((x-v['cen2'])**2 + v['wid2']**2)))) + \
            (np.cos(v['phi_3'])**2)*v['amp3']*(v['wid3']**2/((x-v['cen3'])**2+v['wid3']**2)) + \
            ((np.sin(v['phi_3'])*v['amp3']*v['wid3']*(x-v['cen3']))/((((x-v['cen3'])**2 + v['wid3']**2)))) + \
                (np.cos(v['phi_4'])**2)*v['amp4']*(v['wid4']**2/((x-v['cen4'])**2+v['wid4']**2)) + \
            ((np.sin(v['phi_4'])*v['amp4']*v['wid4']*(x-v['cen4']))/((((x-v['cen4'])**2 + v['wid4']**2))))
    return model - data



''' Main fitting program: Provides up to four Lorentzian fits to the data '''
# Recieves estimated parameters, data, start and end s12 profile index to fit, number of peaks to fit and constraints (% varience allowed between sequential fits)
# Outputs Array of results , Fitted S12Profiles, Labels (headings) of results array , Residuals of the fits
def _fit( params, x, data, field, startProfile, endProfile, peakNumber, constraints):
    resultsArrayHeadings = np.array(['Applied Field (Oe)', 'Y0 (dB)', 'Y0_Error (dB)', 'Linear Gradient','Linear Gradient error' , 'Resonant Frequency_Peak1 (GHz)', 'Resonant Frequency Error_Peak1  (GHz)',\
                       'Resonant Linewidth_Peak1 (GHz)', 'Resonant Linewidth Error_Peak1 (GHz)', 'Resonant Intensity_Peak1 (dB)'\
                       'Resonant Intensity Error_Peak1 (dB)', 'Mixing_Peak1 ', 'Mixing_Peak1 Error','Resonant Frequency_Peak2 (GHz)', 'Resonant Frequency Error_Peak2  (GHz)',\
                       'Resonant Linewidth_Peak2 (GHz)', 'Resonant Linewidth Error_Peak2 (GHz)', 'Resonant Intensity_Peak2  (dB)'\
                       'Resonant Intensity Error_Peak2  (dB)', 'Mixing_Peak2 ', 'Mixing_Peak2 Error' ])
    
    
    
    
    resultsArrayHeadings = np.array(['Applied Field (Oe)', 'Y0 [dB]', 'Y0 Error [dB]' ,\
                            'Gradient','Gradient Error', 'Peak1 Resonant Frequency (GHz)',\
                            'Peak1 Resonant Frequency Error (GHz)', 'Peak1 Resonant Linewidth (GHz)', 'Peak1 Resonant Linewidth Error (GHz)',\
                            'Peak1 Resonant Intensity (dB)', 'Peak1 Resonant Intensity error  (dB)' ,\
                            'Peak1 Resonant Area (dB GHz)', 'Peak1 Resonant Area error (dB GHz)', 'Peak1 Mixing Angle', \
                            'Peak1 Mixing Angle Error', 'Peak2 Resonant Frequency (GHz)', 'Peak2 Resonant Frequency Error (GHz)',\
                            'Peak2 Resonant Linewidth (GHz)', 'Peak2 Resonant Linewidth Error (GHz)',\
                            'Peak2 Resonant Intensity (dB)', 'Peak2 Resonant Intensity error  (dB)' , \
                            'Peak2 Resonant Area (dB GHz)', 'Peak2 Resonant Area Error (dB GHz)', 'Peak2 Mixing Angle', \
                            'Peak2 Mixing Angle Error', 'Peak3 Resonant Frequency (GHz)', 'Peak3 Resonant Frequency Error (GHz)',\
                            'Peak3 Resonant Linewidth (GHz)', 'Peak3 Resonant Linewidth Error (GHz)',\
                            'Peak3 Resonant Intensity (dB)', 'Peak3 Resonant Intensity error  (dB)' , \
                            'Peak3 Resonant Area (dB GHz)', 'Peak3 Resonant Area error (dB GHz)',  'Peak3 Mixing Angle', \
                            'Peak3 Mixing Angle Error',  'Peak4 Resonant Frequency (GHz)', 'Peak4 Resonant Frequency Error (GHz)',\
                            'Peak4 Resonant Linewidth (GHz)', 'Peak4 Resonant Linewidth Error (GHz)',\
                            'Peak4 Resonant Intensity (dB)', 'Peak4 Resonant Intensity error  (dB)' , \
                            'Peak4 Resonant Area (dB GHz)', 'Peak4 Resonant Area error (dB GHz)' , 'Peak4 Mixing Angle', \
                            'Peak4 Mixing Angle Error', 'r Squared'])
    
    

    
    resultsArray = np.zeros(46)
    finalS12Profiles = np.zeros(len(x))
    Residuals  = np.zeros(len(x))
    
    i=0
    while i< (endProfile - startProfile):
        
        
        # do fit, here with the default leastsq algorithm
        minner = Minimizer(LorentzFit, params, fcn_args=(x, data[:,i]), max_nfev=10000)
        result = minner.minimize()

        # calculate final result
        risidual = result.residual
        final = data[:,i] + result.residual
        
        # write error report
        #report_fit(result)
        
        
        ###Find Goodness of Fit###   
        # residual sum of squares
        ss_res = np.sum((data[:,i]- final) ** 2)
        # total sum of squares
        ss_tot = np.sum((data[:,i] - np.mean(data[:,i])) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        
        
        area1, area_err1= _1calcPeakArea(result.params['amp1'].value, result.params['wid1'].value, result.params['amp1'].stderr , result.params['wid1'].stderr, result.params['phi_1'].value,  result.params['phi_1'].stderr )
        area2, area_err2, area3, area_err3, area4, area_err4 =  np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        
        
        if(peakNumber>1):
            if (result.params['amp2'].value==0 or result.params['wid2'].value==0 or result.params['phi_2'].value==0):
                area2, area_err2 = 0, 0 
            else:  
                area2, area_err2= _1calcPeakArea(result.params['amp2'].value,result.params['wid2'].value,result.params['amp2'].stderr, result.params['wid2'].stderr,  result.params['phi_2'].value,  result.params['phi_2'].stderr)
        
        if(peakNumber>2):
            area3, area_err3= _1calcPeakArea(result.params['amp3'].value,result.params['wid3'].value, result.params['amp3'].stderr, result.params['wid3'].stderr, result.params['phi_3'].value,  result.params['phi_3'].stderr)
        
        if(peakNumber>2):
            area4, area_err4= _1calcPeakArea(result.params['amp4'].value,result.params['wid4'].value, result.params['amp4'].stderr, result.params['wid4'].stderr, result.params['phi_4'].value,  result.params['phi_4'].stderr)
        
        
        resFreq1 =  result.params['cen1'].value
        resWid1 =  result.params['wid1'].value
        resAmp1 =  result.params['amp1'].value
        resFreq2 =  result.params['cen2'].value
        resWid2 =  result.params['wid2'].value
        resAmp2 =  result.params['amp2'].value
        
        resultsArray=np.vstack((resultsArray, np.array([field[i], result.params['y0'].value, result.params['y0'].stderr,\
                                result.params['m'].value,result.params['m'].stderr, result.params['cen1'].value,\
                                result.params['cen1'].stderr, result.params['wid1'].value,result.params['wid1'].stderr,\
                                result.params['amp1'].value, result.params['amp1'].stderr, area1, area_err1, result.params['phi_1'].value, \
                                result.params['phi_1'].stderr, result.params['cen2'].value, result.params['cen2'].stderr,\
                                result.params['wid2'].value, result.params['wid2'].stderr, result.params['amp2'].value, \
                                result.params['amp2'].stderr, area2, area_err2, result.params['phi_2'].value, result.params['phi_2'].stderr, \
                                result.params['cen3'].value, result.params['cen3'].stderr, result.params['wid3'].value, \
                                result.params['wid3'].stderr,\
                                result.params['amp3'].value, result.params['amp3'].stderr, area3, area_err3, result.params['phi_3'].value, \
                                result.params['phi_3'].stderr, result.params['cen4'].value, result.params['cen4'].stderr,\
                                result.params['wid4'].value, result.params['wid4'].stderr, result.params['amp4'].value, \
                                result.params['amp4'].stderr, area4, area_err4, result.params['phi_4'].value, result.params['phi_4'].stderr, r2])))
     
        finalS12Profiles = np.vstack((finalS12Profiles, final))
        Residuals = (np.vstack((Residuals, risidual)))
        
        
        
        
        params=result.params
        #print(result.params['mix1'].value)
        #############
        
        print(result.params['cen1'].value)
        ###Update Boundaries###
        
        params['cen1'].set(min=constraints[0]*resFreq1)  # updates lower bound
        params['cen1'].set(max=constraints[1]*resFreq1)  # updates upper bound
        params['wid1'].set(min=constraints[2]*resWid1)  # updates lower bound
        params['wid1'].set(max=constraints[3]*resWid1)  # updates upper bound
        params['amp1'].set(min=constraints[4]*resAmp1)  # updates lower bound
        params['amp1'].set(max=constraints[5]*resAmp1)  # updates upper bound
        params['cen2'].set(min=constraints[0]*resFreq2)  # updates lower bound
        params['cen2'].set(max=constraints[1]*resFreq2)  # updates upper bound
        params['wid2'].set(min=constraints[2]*resWid2)  # updates lower bound
        params['wid2'].set(max=constraints[3]*resWid2)  # updates upper bound
        params['amp2'].set(min=constraints[4]*resAmp2)  # updates lower bound
        params['amp2'].set(max=constraints[5]*resAmp2)  # updates upper bound
        
        
####################

        i+=1
        
    resultsArray = np.delete(resultsArray, (0), axis=0)    
    finalS12Profiles = np.delete(finalS12Profiles, (0), axis=0)
    
    finalS12Profiles = np.transpose(finalS12Profiles)
    Residuals = np.delete(Residuals, (0), axis=0)
    finalResiduals = np.transpose(Residuals)
    return resultsArray, finalS12Profiles, resultsArrayHeadings, finalResiduals



''' Plot fitted s12 profiles (can be a useful sanity check)'''
def _plot(freq, data, finalS12Profiles, risiduals, profileIdx, label,  save  ):

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0) 
    

    ax1.plot(freq, data, "ro")
    ax1.plot(freq, finalS12Profiles, 'k')
    ax2.plot(freq, risiduals, "bo")
    
    ax1.legend(['Data','Fit'], loc='lower right')
    #ax1.legend(['Fit'], loc='lower right')

    ax2.set_xlabel("Frequency (GHz)",family="serif",  fontsize=12)
    ax1.set_ylabel("PSDX",family="serif",  fontsize=12)
    ax2.set_ylabel("Res.",family="serif",  fontsize=12)
    #ax1.legend(['Applied Field ='+ str(int(field[profileIdx]))+     '  Oe'], loc='lower right')
    

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax1.tick_params(axis='x',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='x',which='minor', direction="in", top="on", right="on", bottom="off", length=2, labelsize=8)
    ax1.tick_params(axis='y',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='y',which='minor', direction="in", top="on", right="on", bottom="on", length=2, labelsize=8)

    ax2.tick_params(axis='x',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='x',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)
    ax2.tick_params(axis='y',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='y',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)

    fig.tight_layout()
    if (save == True):
        
        fig.savefig(label+'.png')
        
''' Plot fitted s12 profiles with residuals (can be a useful sanity check) '''
def plotS12DataFitResid(freq, field, data, finalS12Profiles, risiduals, profileIdx, save):

    
    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0) 
    

    ax1.plot(freq, data[:,profileIdx], "ro")
    ax1.plot(freq, finalS12Profiles[:,profileIdx], 'k')
    ax2.plot(freq, risiduals[:,profileIdx], "bo")
    
    ax1.legend(['Data','Fit'], loc='lower right')
    #ax1.legend(['Fit'], loc='lower right')

    ax2.set_xlabel("Frequency (GHz)",family="serif",  fontsize=12)
    ax1.set_ylabel("S12 (dB)",family="serif",  fontsize=12)
    ax2.set_ylabel("Res.",family="serif",  fontsize=12)
    #ax1.legend(['Applied Field ='+ str(int(field[profileIdx]))+     '  Oe'], loc='lower right')
    

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax1.tick_params(axis='x',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='x',which='minor', direction="in", top="on", right="on", bottom="off", length=2, labelsize=8)
    ax1.tick_params(axis='y',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='y',which='minor', direction="in", top="on", right="on", bottom="on", length=2, labelsize=8)

    ax2.tick_params(axis='x',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='x',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)
    ax2.tick_params(axis='y',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='y',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)

    fig.tight_layout()
    if (save == True):
        
        fig.savefig('plot_AppliedField_'+str(field[profileIdx])+'.png')
        
    return



''' Save data to . '''
def _save(data, residuals, finalFits, resultsArray, resultsArrayHeadings,filename ):

    resultFile = np.vstack((resultsArrayHeadings,resultsArray))
    with open(filename+'_fitParams.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(resultFile)
    with open(filename+'_resid.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(residuals)
    with open(filename+'_finalFits.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(finalFits)
    with open(filename+'_dataForFit.txt', 'w') as f:
        csv.writer(f, delimiter=' ').writerows(data)        
          
    return 

def plotS12DataFitResid_All(freq, field, data, finalS12Profiles, risiduals, profileIdx, filename, save):

    #Setup params
    i=0
    #param = [int_Res, width_Res, freq_Res, C, m]
   
    

    while i< (np.shape(finalS12Profiles)[1]):
        fig = plt.figure(figsize=(4,4))
        gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        gs.update(hspace=0) 


        ax1.plot(freq, data[:,i], "ro", markersize=0.1)
        ax1.plot(freq, finalS12Profiles[:,i], 'k--')#,\
        ax2.plot(freq, risiduals[:,i], "bo", markersize=0.1)



        ax2.set_xlabel("Frequency (GHz)",family="serif",  fontsize=12)
        ax1.set_ylabel("S12 (dB)",family="serif",  fontsize=12)
        ax2.set_ylabel("Res.",family="serif",  fontsize=12)

        ax1.legend(loc="best")

        ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
        #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

        ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax1.xaxis.set_major_formatter(plt.NullFormatter())

        ax1.tick_params(axis='x',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
        ax1.tick_params(axis='x',which='minor', direction="out", top="on", right="on", bottom="off", length=5, labelsize=8)
        ax1.tick_params(axis='y',which='major', direction="out", top="on", right="on", bottom="off", length=8, labelsize=8)
        ax1.tick_params(axis='y',which='minor', direction="out", top="on", right="on", bottom="on", length=5, labelsize=8)

        ax2.tick_params(axis='x',which='major', direction="in", top="off", right="on", bottom="on", length=8, labelsize=8)
        ax2.tick_params(axis='x',which='minor', direction="in", top="off", right="on", bottom="on", length=5, labelsize=8)
        ax2.tick_params(axis='y',which='major', direction="in", top="off", right="on", bottom="on", length=8, labelsize=8)
        ax2.tick_params(axis='y',which='minor', direction="in", top="off", right="on", bottom="on", length=5, labelsize=8)

        fig.tight_layout()
        if (save==True):
            fig.savefig(filename+"_"+str(int(field[i]))+"Oe.png", format="png",dpi=1000)
        i+=1
    return


def plotS12DataFitResid(freq, field, data, filename, finalS12Profiles, risiduals, profileIdx, save):

    fig = plt.figure(figsize=(7,7))
    gs = gridspec.GridSpec(2,1, height_ratios=[1,0.25])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    gs.update(hspace=0) 
    

    ax1.plot(freq, data[:,profileIdx], "ro")
    ax1.plot(freq, finalS12Profiles[:,profileIdx], 'k')
    ax2.plot(freq, risiduals[:,profileIdx], "bo")
    
    ax1.legend(['Data','Fit'], loc='lower right')
    #ax1.legend(['Fit'], loc='lower right')

    ax2.set_xlabel("Frequency (GHz)",family="serif",  fontsize=12)
    ax1.set_ylabel("S12 (dB)",family="serif",  fontsize=12)
    ax2.set_ylabel("Res.",family="serif",  fontsize=12)
    #ax1.legend(['Applied Field ='+ str(int(field[profileIdx]))+     '  Oe'], loc='lower right')
    

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))

    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    ax1.tick_params(axis='x',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='x',which='minor', direction="in", top="on", right="on", bottom="off", length=2, labelsize=8)
    ax1.tick_params(axis='y',which='major', direction="in", top="on", right="on", bottom="off", length=4, labelsize=8)
    ax1.tick_params(axis='y',which='minor', direction="in", top="on", right="on", bottom="on", length=2, labelsize=8)

    ax2.tick_params(axis='x',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='x',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)
    ax2.tick_params(axis='y',which='major', direction="in", top="off", right="on", bottom="on", length=4, labelsize=8)
    ax2.tick_params(axis='y',which='minor', direction="in", top="off", right="on", bottom="on", length=2, labelsize=8)

    fig.tight_layout()

    if (save == True):
        
        fig.savefig(filename+'_plot_AppliedField_'+str(field[profileIdx])+'.png')
        
        
        
        
        
    return


''' Fitting of the resonances (To extract magnetic parameters) '''
    
###Equations NB x is external applied H-field
def IPKittel(x, g, Hk, MEff):
    return g**2 * (1.3996E6)**2 * (((abs(x)+Hk) * ( abs(x) + Hk+4*np.pi* MEff)) )

def OOP_KittelFit(x, g, Hk, MEff):
    return 0.000000001*g*(1399600)*(x-4*np.pi*MEff) ###check this

def linewidthFunc(x, g, alpha, Hk, MEff):
    return g * 8.7815587E6 * alpha * (2 * (abs(x)+Hk)+4*np.pi*MEff)





### Fitting functions ###
#removed maxfev=10000
def KittelFitting(x, y, func):
    popt, pcov = curve_fit(func, x, y, maxfev=1000000)
    return popt, pcov

def dampingFit(xdata, ydata, func):
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt, pcov

def linFit(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    return slope, intercept, r_value, p_value, std_err

