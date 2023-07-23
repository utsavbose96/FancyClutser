# -*- coding: utf-8 -*-
"""
Created on Fri May 26 21:08:25 2023

@author: User
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.integrate import quad
import pandas as pd

#%matplotlib notebook  

#constants (cgs):
h_planck = 6.62e-27 #planck constant
c_light = 3.0e10 #speed of light, cm/s
k_boltz = 1.4e-16 #boltzmann constant, erg/K 
s_Stefan = 5.6704e-5  #stefan-boltzmann constant, erg⋅cm−2⋅s−1⋅K−4
(L_sun, r_sun) = (3.85e33, 6.96e10) #Sun luminosity in erg/s, Sun radius in cm
(lamB, lamV, lamR) = (4500., 5600., 6700.) #Wavelengths of different bands, Angstrom


def I(diff_params,y,wavel,I0):
    #gets the intensity of light at a certain position according to diffraction law.
    a,D = diff_params 
    frac=np.pi*a*y/(wavel*D)
    return I0*(np.sin(frac)**2 / frac**2) #intensity at each position y

def gaussian(x, mu, sig): 
    #simple gaussian function
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def bol_corr(Temp, lMin, lMax):
    # calculates the ratio of bolometric luminosity to the luminosity emitted
    # at wavelengths between lMin and lMax (in angstrom)
    (l1, l2) = (lMin*1.e-8, lMax*1.e-8) # angstrom to cm
    bolcorr_list = []
    corr = (s_Stefan* Temp**4) / quad(lambda x: Blam(Temp,x*1e8), l1, l2) 
    return corr[0]

def Blam(T,wavel):
    # blackbody emissivity at wavelength (wavel)
    ll = 1.e-8 * wavel # convert from angstrom to cgs
    bb1 = 2*h_planck*c_light*c_light / ll**5
    bb2 = np.exp(h_planck*c_light/(k_boltz*T*ll))
    return bb1 / (bb2-1.)

def L_to_M(L,T):
    #gets the magnitude from the luminosity and temperature.
    
    # radius from L = (4*pi*R**2) * (s_Stefan*T**4):
    StarRad = (L*L_sun/(4*np.pi*s_Stefan*T**4))**.5 / r_sun
    # luminosity in V band = total luminosity / bolometric correction
    LumV = L / bol_corr(T, lamB, lamR)
    # V magnitude from definition
    magV = -2.5*np.lib.scimath.log10(LumV)

    # compute colors and B and R magnitudes
    (bbB, bbV, bbR) = (Blam(T, lamB), Blam(T, lamV), Blam(T, lamR)) #blackbody emissivities
    B_V = -2.5 * np.lib.scimath.log10(bbB / bbV) #B-V color
    V_R = -2.5 * np.lib.scimath.log10(bbV / bbR) #V-R color
    (magB, magR) = (magV+B_V, magV-V_R) #B and R magnitudes
    return (magV,magB,magR)


def Fb(x,C):  #defines ratio between big and small spikes
    return (x**3/(x**2+C)) 

def plotstar(xstar, ystar, magV, magB, magR,angle,smallspikes,Xc,Yc,Rplot,
             Npix,x,y,diff_params,xpixel,ypixel,C,rescaling,PSF): 
    
    #rescaling luminosities:
    rescaling1,rescaling2=rescaling
    
    LV=10**(-0.4*magV)*rescaling1 +rescaling2  
    LB=10**(-0.4*magB)*rescaling1 +rescaling2  
    LR=10**(-0.4*magR)*rescaling1 +rescaling2  

    #creating frame:
    xx, yy = np.meshgrid(x, y)
    
    allspikesV=np.zeros((Npix,Npix))
    allspikesB=np.zeros((Npix,Npix))
    allspikesR=np.zeros((Npix,Npix))
    
    #calculating distance to star
    r=np.sqrt((xx-xstar)**2 + (yy-ystar)**2)
    
    #getting central intensities for big and small spikes----------------------------------

    Ibig_V= LV/(1+(LV/Fb(LV,C)))
    Ismall_V=Ibig_V*LV/Fb(LV,C)
    
    Ibig_B= LB/(1+(LB/Fb(LB,C)))
    Ismall_B=Ibig_B*LB/Fb(LB,C)
    
    Ibig_R= LR/(1+(LR/Fb(LR,C)))
    Ismall_R=Ibig_R*LR/Fb(LR,C)


    #small spikes--------------------------
    #creating radial gaussian as a function of the luminosity in each band:
    extragaussV=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,0.001*LV)
    extragaussB=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,0.001*LB)
    extragaussR=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,0.001*LR)
    
    #creating a frame for the small spikes in each band:
    ss_smallframeV=np.zeros((Npix,Npix))
    ss_smallframeB=np.zeros((Npix,Npix))
    ss_smallframeR=np.zeros((Npix,Npix))

    #getting diffraction pattern as a function of the distance to the star:
    spikeV=I(diff_params,r,lamV,Ismall_V)
    spikeB=I(diff_params,r,lamB,Ismall_B)
    spikeR=I(diff_params,r,lamR,Ismall_R)
    
    #rescaling the small spikes according to the luminosity in each band:
    ss_bigframeV=smallspikes*LV
    ss_bigframeB=smallspikes*LB
    ss_bigframeR=smallspikes*LR
    
    #adding the small spikes to the frames:
    ss_smallframeV+=ss_bigframeV[int(Npix-ypixel):int(2*Npix-ypixel),int(Npix-xpixel):int(2*Npix-xpixel)] 
    ss_smallframeB+=ss_bigframeB[int(Npix-ypixel):int(2*Npix-ypixel),int(Npix-xpixel):int(2*Npix-xpixel)] 
    ss_smallframeR+=ss_bigframeR[int(Npix-ypixel):int(2*Npix-ypixel),int(Npix-xpixel):int(2*Npix-xpixel)] 
    
    #multiplying everything to get final pattern:
    smallspikeV=spikeV*ss_smallframeV  *extragaussV 
    smallspikeB=spikeB*ss_smallframeB  *extragaussB 
    smallspikeR=spikeR*ss_smallframeR  *extragaussR 
    
    #getting normalization factor:
    normfactorV=np.max(smallspikeV)/Ismall_V
    normfactorB=np.max(smallspikeB)/Ismall_B
    normfactorR=np.max(smallspikeR)/Ismall_R

    #normalizing:
    allspikesV+=smallspikeV/normfactorV #now the center flux is Ismall
    allspikesB+=smallspikeB/normfactorB 
    allspikesR+=smallspikeR/normfactorR 
    
    #big spikes----------------------
    #creating diffraction pattern:
    spikeV=I(diff_params,r,lamV,Ibig_V)
    spikeB=I(diff_params,r,lamB,Ibig_B)
    spikeR=I(diff_params,r,lamR,Ibig_R)
    
    #setting the thickness of the spikes:
    sigmaPSFpix=0.004

    #creating 2 perpendicular spikes:
    dist1=gaussian(np.cos(angle)*(xx-xstar) + np.sin(angle)*(yy-ystar),0,sigmaPSFpix)
    dist2=gaussian(-np.sin(angle)*(xx-xstar) + np.cos(angle)*(yy-ystar),0,sigmaPSFpix)
    
    #multiplying spikes by diffraction pattern:
    bigspikeV=(dist1+dist2)*spikeV
    bigspikeB=(dist1+dist2)*spikeB
    bigspikeR=(dist1+dist2)*spikeR

    #getting normalization factor:
    normfactorV=np.max(bigspikeV)/Ibig_V
    normfactorB=np.max(bigspikeB)/Ibig_B
    normfactorR=np.max(bigspikeR)/Ibig_R

    #normalizing:
    allspikesV+=bigspikeV/normfactorV
    allspikesB+=bigspikeB/normfactorB
    allspikesR+=bigspikeR/normfactorR

    #more concentrated dot for the center of the star---------------------
    #rescaling PSF according to luminosity in each band:
    PSF_V=PSF*LV 
    PSF_B=PSF*LB 
    PSF_R=PSF*LR 
    
    #creating a gaussian with sigma=PSF and scaling with luminosity:
    starV=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,PSF_V) * LV *1.5  
    starB=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,PSF_B) * LB *1.5  
    starR=gaussian((xx-xstar)**2 + (yy-ystar)**2,0,PSF_R) * LR *1.5  
    
    #adding dot to the frame:
    allspikesV+=starV  
    allspikesB+=starB  
    allspikesR+=starR  
    
    return allspikesV, allspikesB, allspikesR

def pixelpos(xstar,ystar,Rplot,Xc,Yc,Npix):  
    #calculates the pixel position corresponding to a given input position.
    xpix,ypix=(Rplot-Xc+xstar)*Npix/(2*Rplot) , (Rplot-Yc+ystar)*Npix/(2*Rplot)
    return np.round(xpix),np.round(ypix)

def gen_smallspikes(Npix,Rplot,Xc,Yc):
    #randomly generates small spikes
    Xc,Yc=0,0
    
    #creating frame twice as big as final frame:
    smallspikes=np.zeros((2*(Npix),2*(Npix)))
    (x2, y2) = (np.linspace(Xc-Rplot, Xc+Rplot, 2*(Npix)), 
              np.linspace(Yc-Rplot, Yc+Rplot, 2*(Npix)))

    xx2, yy2 = np.meshgrid(x2, y2)
    
    #creating 100 spikes with random intensities and adding them to the center of the frame:
    for angle in np.linspace(0,np.pi,180):
        sspike=gaussian(np.cos(angle)*(xx2) + np.sin(angle)*(yy2),0,0.003)*np.random.rand()
        smallspikes+=sspike
        
    return smallspikes


def read_file(filename,simbadfile,compactness): 
    #gets positions and magnitudes from input file
    
    if simbadfile==False:  #using original cluster file
        #reading file:
        print('Reading file...')
        data = np.loadtxt(filename, usecols =(0,1,2,3,4,5,6,7,8,9))
        (Xpos, Ypos) = (data[:,2], data[:,3])
        (Lum, Temp) = (data[:,8], data[:,9])
        
        #getting the magnitude of the star in each band:
        magV=np.zeros(len(Lum))
        magB=np.zeros(len(Lum))
        magR=np.zeros(len(Lum))
        
        #converting luminosities to magnitudes:
        for i,L in enumerate(Lum): 
            magV[i],magB[i],magR[i]=L_to_M(L,Temp[i])
    
        
    else: #Using Simbad data 
        
        #reading file:
        df = pd.read_csv(filename, comment='#',delimiter='|',skiprows=9)
        data = np.array(df)
        data = data[:-1] #deletes last line, which is nan

        #deleting all rows which have none of the 3 magnitudes
        data = np.delete(data,np.where((data[:, 9] == "     ~   ")
                                       &(data[:, 10] == "     ~   ")&(data[:, 11] == "     ~   ")),axis=0)
        #deleting all rows which have no coordinates
        data = np.delete(data,np.where(data[:, 7] == "No Coord.                              "),axis=0)
    
        #asking user whether to simulate missing magnitudes:
        replace_mag=input('Plot stars with missing magnitudes? [y/n]') 
        while replace_mag not in ('y','n'):
            print('Invalid input. Please type "y" or "n".')
            replace_mag=input('Plot stars with missing magnitudes? [y/n]')
            
        print('Reading file...')
        if replace_mag=='y':
            #calculating colours:
            cond=(data[:,10]!= "     ~   ")&(data[:,9]!= "     ~   ")
            VB=(data[:,10][np.where(cond)]).astype(float)-(data[:,9][np.where(cond)]).astype(float)#V-B
            cond=(data[:,11]!= "     ~   ")&(data[:,10]!= "     ~   ")
            RV=(data[:,11][np.where(cond)]).astype(float)-(data[:,10][np.where(cond)]).astype(float)#R-V
            cond=(data[:,11]!= "     ~   ")&(data[:,9]!= "     ~   ")
            RB=(data[:,11][np.where(cond)]).astype(float)-(data[:,9][np.where(cond)]).astype(float)#R-B
            
            #replacing missing magnitudes:
            for i,row in enumerate(data):
                if (row[10]=="     ~   ")&(row[11]=="     ~   ")&(row[9]!="     ~   "):
                    data[i,10]=float(row[9])+np.mean(VB)
                    data[i,11]=float(row[9])+np.mean(RB)
                if (row[9]=="     ~   ")&(row[11]=="     ~   ")&(row[10]!="     ~   "):
                    data[i,9]=float(row[10])-np.mean(VB)
                    data[i,11]=float(row[10])+np.mean(RV)
                if (row[10]=="     ~   ")&(row[9]=="     ~   ")&(row[11]!="     ~   "):
                    data[i,9]=float(row[11])-np.mean(RB)
                    data[i,10]=float(row[11])-np.mean(RV)

                if (row[9]=="     ~   ")&(row[10]!="     ~   ")&(row[11]!="     ~   "):
                    c1=float(row[11])-np.mean(VB)
                    c2=float(row[10])-np.mean(RB)
                    data[i,9]=(c1+c2)/2
                if (row[10]=="     ~   ")&(row[9]!="     ~   ")&(row[11]!="     ~   "):
                    c1=float(row[9])+np.mean(VB)
                    c2=float(row[11])-np.mean(RV)
                    data[i,10]=(c1+c2)/2
                if (row[11]=="     ~   ")&(row[10]!="     ~   ")&(row[9]!="     ~   "):
                    data[i,11]=float(row[10])+np.mean(RV)
                    c1=float(row[9])+np.mean(RB)
                    c2=float(row[10])+np.mean(RV)
                    data[i,11]=(c1+c2)/2                    
                        
        elif replace_mag=='n':
            #deleting rows with missing magnitudes:
            data = data[(data[:, 9] != "     ~   ")&(data[:, 10] != "     ~   ")&(data[:, 11] != "     ~   ")]
            
        #Storing the magnitudes
        magB = data[:, 9].astype(float) 
        magV = data[:, 10].astype(float) 
        magR = data[:, 11].astype(float) 

        #converting positions from ra and dec to arcseconds:
        pos = data[:,7]
        
        x_arc = []
        y_arc = []
        for p in pos:
            ra_d,ra_m,ra_s, dec_d, dec_m,dec_s = p.split()

            # Convert RA to arcsec
            x_arc.append((float(ra_d)*3600 + float(ra_m)*60 + float(ra_s)) * 15)

            # Convert Dec to arcsec
            y_arc.append(float(dec_d)*3600 + float(dec_m)*60 + float(dec_s))


        #converting to coordinates with axis at cluster centre
        x_mean = np.mean(x_arc)
        y_mean = np.mean(y_arc)
        Xpos= (x_arc-x_mean) / compactness
        Ypos = (y_arc-y_mean) / compactness
    print('...done')
    print('Total number of stars:', len(data))
    return Xpos, Ypos, magV, magB, magR


    
def makeimg(filename, simbadfile=False,cpos=(0,0),angle=0,Rplot=1,savefig=True,
                output_file="fancyfig.png",Npix=1000,diff_params=(800,0.1),C=100,
            rescaling=(0.01,0.3),compactness=1000,PSF=0.00003):    
    
    #reading data:
    Xpos, Ypos, magV, magB, magR = read_file(filename,simbadfile,compactness)
    print('Initializing...')

    Xc,Yc=cpos
    
    #creating a frame with small spikes
    smallspikes=gen_smallspikes(Npix,Rplot,Xc,Yc)
    
    #creating axes
    (x, y) = (np.linspace(Xc-Rplot, Xc+Rplot, Npix), 
                np.linspace(Yc-Rplot, Yc+Rplot, Npix))
    
    #empty frames where stars will be added
    frameV,frameR,frameB=np.zeros((Npix,Npix)),np.zeros((Npix,Npix)),np.zeros((Npix,Npix))
    
    #getting pixel positions of the stars
    Xpix,Ypix= pixelpos(Xpos, Ypos,Rplot,Xc,Yc,Npix)

    inview=0
    for xstar, ystar, magV, magB, magR,xpixel,ypixel in tqdm(zip(Xpos, Ypos, magV, magB, magR,Xpix,Ypix),position=0, leave=True):
        
        #checking whether the star is in the frame
        if xstar<x[0] or xstar>x[-1] or ystar<y[0] or ystar>y[-1]:
            continue
        inview+=1
        
        #adding star to each frame
        starV,starB,starR=plotstar(xstar, ystar, magV, magB, magR,angle,smallspikes,Xc,Yc,Rplot,Npix,
                                   x,y,diff_params,xpixel,ypixel,C,rescaling,PSF)
        frameV+=starV
        frameB+=starB
        frameR+=starR
    #setting array with 3 frames for plotting
    imgRGB_tmp = np.array([frameR, frameV, frameB ]) #V acts as G in RGB
    imgRGB = np.rollaxis(imgRGB_tmp, 0, 3) 


    print('Stars in view:', inview, 'out of',len(Xpos))
    
    plt.figure()
    plt.axis('off')
    plt.imshow(imgRGB)
    if savefig==True:
        plt.savefig(output_file)
    plt.show()

#M4:
#best parameters for plotting full cluster:
#makeimg(filename='sim-m4.txt',simbadfile=True,Rplot=0.5,diff_params=(5000,0.1),
        #output_file='fancy_fig_m4_full.png',rescaling=(8000,0.001),C=0.5) 

#best parameters for plotting only the magnitudes we have:
# makeimg(filename='sim-m4.txt',simbadfile=True,Rplot=0.5,diff_params=(5000,0.1),
#         output_file='fancy_fig_m4.png',rescaling=(8000,0.2),C=1) 

#Original file:
#makeimg(filename="one_star_data.txt",Rplot=0.8,angle=0,output_file='Mapelli_data.png')