# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:03:32 2021

Beam optimizer module.

This module implements the genetic algorithm for beam mode frequency design
described in "Multi-modal Tuning of Vibrating Bars with Simplified Undercuts
Using an Evolutionary Optimization Algorithm" by F. Soares, J. Antunes, and 
V. Debut in the journal Applied Acoustics, published online October 21st 2020.

The module "beamanalysis" is used for computing beam mode frequencies and mode shapes.

@author: Russell Hawkins
"""


import numpy as np
import copy
import beamanalysis as ba


def fitnessSortFunc(e):
    
    # This function is returns the element associated with the key 'fitness' 
    # associated with the input dictionary "e". Used to sort the population by
    # fitness.
    
    return e['fitness']


def completeLengths(Lhalf, Ltotal):
    
    # This function converts the geometric form used in optimization to the form 
    # used for the beam analysis, the segment lengths. See figure 3 in Soares et. al.
    # for a depiction of the geometry of the beam.

    L = np.concatenate((Lhalf, np.array([Ltotal/2 - np.sum(Lhalf)]))); # Add last segment
    
    L[0] = 2*L[0]; #Double length of middle segment, due to symmetry
        
    L = np.concatenate((np.flip(L[1:]), L)); # Adjoin mirrored segments
    
    return L


def completeThicknesses(hhalf, hmax):
    
    # This function converts the geometric form used in optimization to the form 
    # used for the beam analysis, the segment thicknesses. See figure 3 in Soares et. al.
    # for a depiction of the geometry of the beam.
        
    h = np.concatenate((hhalf, np.array([hmax]))); #Add end chunk, with thickness of uncut bar
    
    h = np.concatenate((np.flip(h[1:]), h)); #  Adjoin mirrored segments
    
    return h


def popinitializer(Npop, Ncuts, Nmodes, Lhalfinit, hhalfinit, Ltotal, hmax):
    
    # This function initializes the population. It produces a population of Npop 
    # individuals that each have identical lengths and thicknesses of segments,
    # defined by Lhalfinit and hhalfinit.
    
    population = [];

    for i in range(Npop):
    
        
        # Get the complete lengths
        L = completeLengths(Lhalfinit, Ltotal);
        
        # Get complete thicknesses
        h = completeThicknesses(hhalfinit, hmax)
        
        # Add a spot for the mode frequencies
        modefreqs = np.zeros(Nmodes);
        
        # Add a spot for the fitness function
        fitness = 0;
        
        # Add a spot for the error in cents
        centserror = np.zeros(Nmodes);
        
        # Put it into a dict
        # My population is a population of bugs!
        bug = {'halflengths': Lhalfinit, 'halfthicknesses': hhalfinit, 'completelengths': L, 'completethicknesses': h, 'modefreqs': modefreqs, 'fitness': fitness, 'centserror': centserror}; 
    
        # Add bug to population
        population.append(bug)
        
    return population



def evaluateFitness(rho, k, E, G, w, population, targetfreqs, Nmodes):
    
    # This function evaluates the fitness of a population. For each member, it
    # computes a scaled mean squared error between its mode frequencies and the 
    # target mode frequencies. This error is what we are trying to minimize.
    
    # Incidentally this probably shouldn't be called fitness, because we're minimizing
    # it, whereas biologial evolution, which inspires this technique, maximizes fitness,
    # but that is neither her nor there.
    
    for bug in population:
        
        # Get complete length vector
        L = bug['completelengths'];
        
        # Get complete thicknesses vector
        h = bug['completethicknesses'];
        
        # Compute the mode frequencies
        modefreqs = ba.beammodefreq(rho, k, E, G, w, h, L, 100, 0.001, Nmodes);
        bug['modefreqs'] = modefreqs;
        
        # Compute fitness, eq. 7 in Soares et. al.
        fitness = (100/Nmodes) * np.sum(((modefreqs - targetfreqs)/ targetfreqs)**2);
        bug['fitness'] = fitness;
        
        # Compute cents error
        bug['centserror'] = 1200*np.log2(modefreqs/targetfreqs);

    return population


def mutate(toMutate, Ncuts, Ltotal, Lmin, hmin, hmax, mutstrength):
    
    # This function mutates all of the members of the population toMutate.
    # Changes in lengths and thicknesses are made at random, while staying
    # within min and max thickness and maintaining total length fixed.
    
    # See Soares et. al. section 3.3 for detailed description, this is the uniform
    # random operator approach, not the self-adaptive Gaussian.
    
    for bug in toMutate:
        
        # Generate number of mutations to make
        nummut = np.random.randint(2*Ncuts); 
    
        # Generate nummut integers 
        muts = np.random.choice(2*Ncuts, size = nummut, replace=False); # Generate nummut integers 
        
        r = np.random.uniform(-1,1, size = nummut);
    
        for i in range(len(muts)):
            
            if muts[i] <= Ncuts - 1:
                
                oldlength = bug['halflengths'][muts[i]];
                
                newlength = oldlength + r[i] * mutstrength * (Ltotal/2);
                
                if newlength < Lmin:
                    
                    # Enforce that segment lengths must be greater than or equal to Lmin
                    newlength = Lmin;
                
                bug['halflengths'][muts[i]] = newlength;
                
            else:
                
                oldthickness = bug['halfthicknesses'][muts[i] - Ncuts];
                
                newthickness = oldthickness + r[i] * mutstrength * hmax;
                
                if newthickness < hmin:
                    
                    #Enforce min thickness constraint
                    newthickness = hmin;
                
                if newthickness > hmax:
                    
                    # Enforce max thickness constraint
                    newthickness = hmax;
                
                bug['halfthicknesses'][muts[i] - Ncuts] = newthickness;
                
            #Enforce length constraint
            Lhalf = bug['halflengths'];
        
            Lhalf = np.concatenate((Lhalf, np.array([Ltotal/2 - np.sum(Lhalf)]))); # Add last segment
    
            #Scale L back to Ltotal
            Lhalf = ((Ltotal/2) / np.sum(Lhalf)) * Lhalf;
            
            # Enforce Lmin constraint on last segment
            if Lhalf[-1] < Lmin:
                
                # Rescale the cut lengths so that they sum to Ltotal/2 - Lmin
                # The last segment, which has the thickness of the uncut bar, will have length Lmin
                Lhalf[0:-1] = ((Ltotal/2 - Lmin) / np.sum(Lhalf[0:-1])) * Lhalf[0:-1]
                
            
            bug['halflengths'] = Lhalf[0:-1];
            
            #Recompute complete lengths and thicknesses
            L = completeLengths(bug['halflengths'], Ltotal);
            h = completeThicknesses(bug['halfthicknesses'], hmax);
            
            bug['completelengths'] = L;
            bug['completethicknesses'] = h;


    return toMutate


def roulette(population, Npop, Nc):
    
    # This function is a helper function for the Crossover function. It generates
    # a set of indices representing pairs to be "mated", that is, to have their
    # segment lengths and thicknesses combined to produce "offspring". See section
    # 3.1 of Soares et. al.         
        
    sumInverseFitness = 0; # Initialize sum of inverse fitness
    
    for bug in population:
        
        sumInverseFitness += 1 / bug['fitness'];
        
    ps = np.zeros(Npop)
    
    for i in range(Npop):
        
        ps[i] = (1/sumInverseFitness) * (1/population[i]['fitness']);
        
    r = np.random.random(Nc); # Generate random floats [0,1)
    
    mateIndices = np.digitize(r, np.cumsum(ps));    
    
    return mateIndices
        

def crossover(population, Ltotal, Lmin, hmin, hmax, Npop, Nc):
    
    # This function performs crossover on population members chosen stochastically
    # from the population by means of the Roulette function. Segment lengths and 
    # thicknesses are combined to produce "offspring", while respecting the various
    # geometric constraints of minimum segment thickness and length, maximum thickness,
    # and fixed total length.
    
    mateIndices = roulette(population, Npop, Nc)
    
    children = [];
    
    for i in np.arange(0, Nc, 2):
        
        L1 = population[mateIndices[i]]['halflengths'];
        L2 = population[mateIndices[i+1]]['halflengths'];
        
        h1 = population[mateIndices[i]]['halfthicknesses'];
        h2 = population[mateIndices[i+1]]['halfthicknesses'];
    
        r = np.random.random();
        
        # Generate children
        L1child = L1 + r * (L2 - L1);
        L2child = L2 + r * (L1 - L2);
        
        h1child = h1 + r * (h2 - h1);
        h2child = h2 + r * (h1 - h2);
        
        # Enforce min/max thickness constraint
        
        if np.any(h1child < hmin):
            
            h1child[np.where(h1child < hmin)] = hmin;
            
        if np.any(h2child < hmin):
            
            h2child[np.where(h2child < hmin)] = hmin;
            
        if np.any(h1child > hmax):
            
            h1child[np.where(h1child > hmax)] = hmax;
            
        if np.any(h2child > hmax):
            
            h2child[np.where(h2child > hmax)] = hmax;
            
        # Rescale lengths to maintain constant total length
        Lhalf1 = L1child;
        
        Lhalf1 = np.concatenate((Lhalf1, np.array([Ltotal/2 - np.sum(Lhalf1)]))); # Add last segment
    
        #Scale L back to Ltotal
        Lhalf1 = ((Ltotal/2) / np.sum(Lhalf1)) * Lhalf1;
        
        # Enforce Lmin constraint on last segment
        if Lhalf1[-1] < Lmin:
                
            # Rescale the cut lengths so that they sum to Ltotal/2 - Lmin
            # The last segment, which has the thickness of the uncut bar, will have length Lmin
            Lhalf1[0:-1] = ((Ltotal/2 - Lmin) / np.sum(Lhalf1[0:-1])) * Lhalf1[0:-1]
            
        L1child = Lhalf1[0:-1];
        
        Lhalf2 = L2child;
        
        Lhalf2 = np.concatenate((Lhalf2, np.array([Ltotal/2 - np.sum(Lhalf2)]))); # Add last segment
    
        #Scale L back to Ltotal
        Lhalf2 = ((Ltotal/2) / np.sum(Lhalf2)) * Lhalf2;
        
        # Enforce Lmin constraint on last segment
        if Lhalf2[-1] < Lmin:
                
            # Rescale the cut lengths so that they sum to Ltotal/2 - Lmin
            # The last segment, which has the thickness of the uncut bar, will have length Lmin
            Lhalf2[0:-1] = ((Ltotal/2 - Lmin) / np.sum(Lhalf2[0:-1])) * Lhalf2[0:-1]
            
        L2child = Lhalf2[0:-1];
        
        
        # Update children
        population[mateIndices[i]]['halflengths'] = L1child;
        population[mateIndices[i+1]]['halflengths'] = L2child;
        population[mateIndices[i]]['halfthicknesses'] = h1child;
        population[mateIndices[i+1]]['halfthicknesses'] = h2child;
        
        # Update children's complete vectors
        population[mateIndices[i]]['completelengths'] = completeLengths( L1child, Ltotal);
        population[mateIndices[i+1]]['completelengths'] = completeLengths( L2child, Ltotal);
        population[mateIndices[i]]['completethicknesses'] = completeThicknesses( h1child, hmax);
        population[mateIndices[i+1]]['completethicknesses'] = completeThicknesses( h2child, hmax);
            
        # Compile list of children
        children.append(population[mateIndices[i]]);
        children.append(population[mateIndices[i + 1]]);
        
    return children


def evolve(rho, k, E, G, w, population, Npop, Ngen, Ne, Nm, Nc, mutstrength, Ncuts, Ltotal, Lmin, hmin, hmax, targetfreqs, Nmodes):
    
    # This function combines the functionalities of the various functions above
    # and implements a complete optimization run. A population of size Npop is
    # evaluated, then the members of the population are selected, mutated, and 
    # crossed over to create the next generation, which by virtue of these rules
    # will have better fitness. This is repeated for Ngen generations.
    
    # Evolve the population
    bestofeachgen = np.zeros(Ngen);
    
    for i in range(Ngen):
            
        # Sort by fitness   
        population.sort(key = fitnessSortFunc);
        
        # Record best fitness
        bestofeachgen[i] = np.mean(np.abs(population[0]['centserror']));
        
        # Select fittest Ne individuals (pass them to next generation unchanged)
        elite = copy.deepcopy(population[0:Ne]);
        
        # Mutate next Nm individuals
        toMutate = copy.deepcopy(population[10: Nm + 10]);
        #toMutate = copy.deepcopy(2*population[0:Nm]); #Mutate top 20 inviduals, but doubled so that total pop is 50
        # Here we are selecting out the bottom 30 out of 50
        
        mutated = mutate(toMutate, Ncuts, Ltotal, Lmin, hmin, hmax, mutstrength);
        
        # For the rest, we mate them
        children = crossover(population, Ltotal, Lmin, hmin, hmax, Npop, Nc)
        
        # Generate new population
        population = [];
            
        population = elite + mutated + children;
        #population = elite + mutated
        
        # Evaluate fitness    
        population = evaluateFitness(rho, k, E, G, w, population, targetfreqs, Nmodes);
        
        print('Generation {0} of {1} complete!'.format(i+1, Ngen))
        
        
    # Sort final population, puts the best at the start 
        
    population.sort(key = fitnessSortFunc);
    
    return bestofeachgen, population












