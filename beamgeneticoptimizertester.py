# -*- coding: utf-8 -*-
"""
Created on Thu May 13 08:20:26 2021

This script implements the genetic optimization algorithm descibed in Soares et. al. 2020.
Script assumes there is a file titled "Results" saved in its file.

@author: hawki

"""


import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import beamanalysis as ba
import beamgeneticoptimizer as bgo



#Beam material parameters, starting with Ansys structural steel

# rho = 7850; #kg/m^3 for Ansys Structural Steel

# nu = 0.3 # Poisson ratio

# k = 5*(1 + nu)/(6 + 5*nu);

# E = 2e11; #Young's modulus / modulus of elasticity

# G = 7.6923e10; #Shear modulus
    
# Aluminum! Material properties from Ansys
rho = 2770; #kg/m^3

nu = 0.33 # Poisson ratio

k = 5*(1 + nu)/(6 + 5*nu);

E = 7.1e10; #Young's modulus / modulus of elasticity

G = 2.6692e10; #Shear modulus

w = 0.05;


# Optimization parameters
# Grabbed from section 4 of Soares et. al.

Ngen = 10; # Number of generations

Ne = 5; # Elitism number, fittest fraction of population, going unchanged into next generation.

Nm = 19; # Mutation number

Nc = 26; # Crossover number, needs to be even!

Npop = Ne + Nm + Nc;

mutstrength = 0.02; # Mutation strength


# Bar geometry constraints

Ltotal = 0.35; # Total length of bar, fixed

Lmin = 0.005; # Segment length minimum

hmin = 0.003; # Minimum allowed thickness for any bar segment

hmax = 0.01; # Max thickness, thickness of uncut bar.

Ncuts = 2; # Number of cuts, due to symmetry this translates to 2*Ncuts+1 total segments



##############################################################################
# Target frequencies

targetfreqs = 175*np.array([1,4,10]); 

targetnote = 'f3'; # Musical note corresponding to this target fundamental frequency.

Nmodes = len(targetfreqs);

##############################################################################



#Let's do this

#Initialize the population

# Initialize with uniform segment lengths and thicknesses
Lhalfinit = ((Ltotal/2) / (Ncuts + 1)) * np.ones(Ncuts); 
hhalfinit = (hmax/2)*np.ones(Ncuts);

# Good initial parameters for 350mm f3 bar with 2 cuts
# Lhalfinit = np.array([0.05236692, 0.04085965]);
# hhalfinit = np.array([0.00447285, 0.00822068]);

population = bgo.popinitializer(Npop, Ncuts, Nmodes, Lhalfinit, hhalfinit, Ltotal, hmax)

#Initialize fitness
population = bgo.evaluateFitness(rho, k, E, G, w, population, targetfreqs, Nmodes);
    

# Evolve the population
bestofeachgen, population = bgo.evolve(rho, k, E, G, w, population, Npop, Ngen, Ne, Nm, Nc, mutstrength, Ncuts, Ltotal, Lmin, hmin, hmax, targetfreqs, Nmodes);


# Plot 1st mode of best solution
h = population[0]['completethicknesses']
L = population[0]['completelengths']
modefreqs = population[0]['modefreqs']

ba.modeplotter(rho, k, E, G, w, h, L, modefreqs[0]);


# Save results!

# Get absolute path to this folder
pathPythonCode = os.getcwd()

# Create path to Results folder
abspathResults = pathPythonCode + '\\Results\\';

# Get the date and time 
now = datetime.now()
nowstr = now.strftime('%m-%d-%Y_%H-%M')

# Path for this run's folder
pathThisRun = abspathResults + '\\AlBeamDesign' + targetnote + 'MeanError{0}Cents'.format(np.mean(np.abs(population[0]['centserror'])).astype('int')) + nowstr

# Make directory named SimpleRNNRun_nowstr
os.mkdir(pathThisRun)

# Switch to the new directory
os.chdir(pathThisRun)

# Make a csv containing all the physical parameters 

# Make dicts
physicalParams = {'Material': 'Aluminum', 'density': rho, 'shear coefficient': k, 'elastic modulus': E, 'shear modulus': G, 'width': w, 'total length': Ltotal};

optimizerParams = {'number of generations': Ngen, 'Elite number': Ne, 'Mutate number': Nm, 'Crossover number': Nc, 'Total Pop': Npop, 'mutation strength': mutstrength, 'number of cuts': Ncuts, 'total length': Ltotal, 'min segment length': Lmin, 'thickness minimum': hmin, 'thickness max': hmax, 'target note': targetnote, 'target frequencies': targetfreqs};

results = {'cents error': population[0]['centserror'], 'segment lengths': L, 'segment thicknesses': h}


# Write parameters and results to csv
f = open('physicalParams.csv', 'w')
w = csv.writer(f)
for key, val in physicalParams.items():
    w.writerow([key, val])
    
f.close()
    
f = open('optimizerParams.csv', 'w')
w = csv.writer(f)
for key, val in optimizerParams.items():
    w.writerow([key, val])
    
f.close()
    
f = open('results.csv', 'w')
w = csv.writer(f)
for key, val in results.items():
    w.writerow([key, val])
    
f.close()


#  Change directory back to original directory, PythonCode
os.chdir(pathPythonCode)

#Plot fitness over generations
plt.plot(bestofeachgen);
plt.show()

# Print final summary
print('')
print('The best solution found has mode frequencies:')
print(population[0]['modefreqs'])
print('This equates to errors of:')
print('{0} cents'.format(population[0]['centserror']))
print('')
print('The best solution has segment lengths:')
print(population[0]['completelengths'])
print('and segment thicknesses:')
print(population[0]['completethicknesses'])


















