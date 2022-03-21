# By Liam Boutin

# Low temperature shift reactor modeling Mk 3. Working version of earlier Mk 1. Solves reactor's governing ODEs to find outlet component
# molar fraction as a function of catalyst mass.

#Kinetic data was obtained from the 2010 review of WGSR kinetics by Smith et al; citation below:
#Smith, Byron & Muruganandam, L. & Murthy, Loganathan & Shantha, Shekhar. (2010). A Review of the Water Gas Shift Reaction Kinetics. International Journal of Chemical Reactor Engineering - INT J CHEM REACT ENG. 8. 10.2202/1542-6580.2238. The world's progression towards the Hydrogen economy is facilitating the production of hydrogen from various resources. In the carbon based hydrogen production, Water gas shift reaction is the intermediate step used for hydrogen enrichment and CO reduction in the synthesis gas. This paper makes a critical review of the developments in the modeling approaches of the reaction for use in designing and simulating the water gas shift reactor. Considering the fact that the rate of the reaction is dependent on various parameters including the composition of the catalyst, the active surface and structure of the catalyst, the size of the catalyst, age of the catalyst, its operating temperature and pressure and the composition of the gases, it is difficult to narrow down the expression for the shift reaction. With different authors conducting experiments still to validate the kinetic expressions for the shift reaction, continuous research on different composition and new catalysts are also reported periodically. Moreover the commercial catalyst manufacturers seldom provide information on the catalyst. This makes the task of designers difficult to model the shift reaction. This review provides a consolidated listing of the various important kinetic expressions published for both the high temperature and the low temperature water gas shift reaction along with the details of the catalysts and the operating conditions at which they have been validated.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Inlet flow rates

f0CO = 291.4 #inlet molar flow of CO, mol/s
f0H2O = 5441.7 #inlet molar flow of H2O, mol/s
f0CO2 = 1604.2 #inlet molar flow of CO2, mol/s
f0H2 = 5015.6 #inlet molar flow of H2, mol/s

InletFlow=[f0CO,f0H2O,f0CO2,f0H2]
fTotal=sum(InletFlow)

#Parameters, universal constants, and reactor config
T = 497.15 #Op. temp, Kelvin
P = 2050000 #Op. pressure, pascals
R = 8.314 #gas constant, base SI units (J/K*mol)

wMax = 3000 #Max range of catalyst mass, kg
wMin = 0 #Min range of catalyst mass, kg

#Kinetic parameters
a = 0.47; #from Smith et al
b = 0.72; #from Smith et al
c = -0.65; #from Smith et al
d = -0.38; #from Smith et al

lonA = 19.25
A = np.exp(lonA) #mol/g/h/atm^-0.16
newA = A*1000/3600*101325**(-0.16)*P**0.16 #mol/kgcat/s
Ea = 79.2e3 # J/mol
k = newA*np.exp(-Ea/(R*T)) #A*exp(-Ea/R/T)
Ke = np.exp(4577.8/T-4.33); # from literature (Smith et al) - unitless

#Define system of ODEs
def LTS(z,W):
    
    #Concentration
    cTotal=P/(R*T) #total concentration
    
    [FCO2O,FH2O,FCO2,FH2]=z
    
    Ft=FCO2O+FH2O+FCO2+FH2
    c_CO=cTotal*FCO2O/Ft #CO
    c_H2O=cTotal*FH2O/Ft #H2O
    c_CO2=cTotal*FCO2/Ft #CO2
    c_H2=cTotal*FH2/Ft #H2
    
    #Rate law; see Smith et al
    beta=(c_H2*c_CO2/(c_CO*c_H2O)/Ke);
    rate=k*(c_CO**a)*(c_H2O**b)*(c_CO2**d)*(c_H2**c)*(1-beta); #kmol/h-gcat
    
    rateCO=-rate
    rateH2O=-rate
    rateCO2=rate
    rateH2=rate

    dCOdW = rateCO
    dH2OdW = rateH2O
    dCO2dW = rateCO2
    dH2dW = rateH2
    
    return dCOdW, dH2OdW, dCO2dW, dH2dW

#Solve system of ODEs
wSpan = np.linspace(wMin, wMax, num=wMax)
solver = odeint(LTS,InletFlow,wSpan)

# Plot results
plt.plot(wSpan,solver[:,0], label='F_CO')
plt.plot(wSpan,solver[:,1], label='F_H2O')
plt.plot(wSpan,solver[:,2], label='F_CO2')
plt.plot(wSpan,solver[:,3], label='F_H2')
plt.xlabel('Catalyst mass, kg')
plt.ylabel('Molar Flowrates (mol/s)')
plt.title('LTS Molar Flow as Function of Cat. Mass')
plt.legend(loc = 'best')
