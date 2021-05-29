# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:48:35 2021

Beam analysis module.

This module serves to compute the mode frequencies of a Timoshenko beam with
rectangular cross-section and piecewise constant thickness. The method is based 
entirely on that described in "Basic Physics of Xylophone and Marimba Bars" by 
B. H. Suits in the American Journal of Physics (2000), with some help from the paper
"The Effect of Rotary Inertia and of Shear Deformation on the Frequency and Normal
Mode Equations of Uniform Beams With Simple End Conditions" by T.C. Huang in the
Journal of Applied Mathematics (1961). One important bit of help being that the 
alpha and beta equations in equation (9) of Suits is incorrect, and instead equation (23)
from Huang should be used.

@author: Russell Hawkins 
"""

import numpy as np
import matplotlib.pyplot as plt


def modematrix(rho, k, E, G, w, h, L, omega):
    
    # This function takes in the material and geometrical parameters of our beam
    # and computes a frequency dependent matrix. The frequencies at which the 
    # determinant of this matrix are zero are our mode frequencies.
    
    
    #Compute a whole helluvalot of constants
    
    N = len(h); #Number of elements
    S = w * h; #Beam cross-sectional areas
    I = w * h**3 / 12; #Cross-section area moments of inertia    
    
    # b, r, and s parameters
    b = np.sqrt(rho * S * L**4 * omega**2 / (E * I));
    r = np.sqrt(I / (S * L**2));
    s = np.sqrt((E / (k * G)) * r**2);
    
    #Correct alpha and beta definitions from Huang '61
    alpha = np.sqrt(1/2) * np.sqrt(-(r**2 + s**2) + np.sqrt((r**2 - s**2)**2 + 4/b**2));
    beta = np.sqrt(1/2) * np.sqrt(r**2 + s**2 + np.sqrt((r**2 - s**2)**2 + 4/b**2));
    
    #Conversion factors for converting the primed into unprimed C's, see eqs. 31-34 of Huang
    deprime1 = (b/L) * ((alpha**2 + s**2) / alpha); 
    deprime2 = (b/L) * ((alpha**2 + s**2) / alpha);
    deprime3 = -(b/L) * ((beta**2 - s**2) / beta);
    deprime4 = (b/L) * ((beta**2 - s**2) / beta);
    
    
    #Construct the matrix
    
    matrix = np.zeros((4*N,4*N));
    
    #Left boundary conditions, 2 equations
    matrix[0,0] = b[0]*alpha[0]*deprime1[0]; 
    matrix[0,2] = b[0]*beta[0]*deprime3[0];
    
    matrix[1,1] = b[0]*alpha[0] - L[0]*deprime2[0]; 
    matrix[1,3] = b[0]*beta[0] - L[0]*deprime4[0];
    
    #Interface conditions!
    
    for i in range(1,N):
    
        #Continuity of Y
        matrix[4*i - 2, 4*(i-1)] = np.cosh(b[i-1]*alpha[i-1]); 
        matrix[4*i - 2, 4*(i-1) + 1] = np.sinh(b[i-1]*alpha[i-1]); 
        matrix[4*i - 2, 4*(i-1) + 2] = np.cos(b[i-1]*beta[i-1]); 
        matrix[4*i - 2, 4*(i-1) + 3] = np.sin(b[i-1]*beta[i-1]);
        matrix[4*i - 2, 4*(i-1) + 4] = -1; 
        matrix[4*i - 2, 4*(i-1) + 6] = -1;
        
        #Continuity of Psi
        matrix[4*i - 1, 4*(i-1)] = np.sinh(b[i-1]*alpha[i-1])*deprime1[i-1]; 
        matrix[4*i - 1, 4*(i-1) + 1] = np.cosh(b[i-1]*alpha[i-1])*deprime2[i-1];
        matrix[4*i - 1, 4*(i-1) + 2] = np.sin(b[i-1]*beta[i-1])*deprime3[i-1]; 
        matrix[4*i - 1, 4*(i-1) + 3] = np.cos(b[i-1]*beta[i-1])*deprime4[i-1];
        matrix[4*i - 1, 4*(i-1) + 5] = -deprime2[i];
        matrix[4*i - 1, 4*(i-1) + 7] = -deprime4[i];
        
        #Continuity of Moment
        matrix[4*i, 4*(i-1)] = (I[i-1]/L[i-1])*b[i-1]*alpha[i-1]*np.cosh(b[i-1]*alpha[i-1])*deprime1[i-1];
        matrix[4*i, 4*(i-1) + 1] = (I[i-1]/L[i-1])*b[i-1]*alpha[i-1]*np.sinh(b[i-1]*alpha[i-1])*deprime2[i-1];
        matrix[4*i, 4*(i-1) + 2] = (I[i-1]/L[i-1])*b[i-1]*beta[i-1]*np.cos(b[i-1]*beta[i-1])*deprime3[i-1];
        matrix[4*i, 4*(i-1) + 3] = -(I[i-1]/L[i-1])*b[i-1]*beta[i-1]*np.sin(b[i-1]*beta[i-1])*deprime4[i-1];
        matrix[4*i, 4*(i-1) + 4] = -(I[i]/L[i])*b[i]*alpha[i]*deprime1[i];
        matrix[4*i, 4*(i-1) + 6] = -(I[i]/L[i])*b[i]*beta[i]*deprime3[i];
        
        #Continuity of Shear
        matrix[4*i + 1, 4*(i-1)] = (S[i-1]/L[i-1])*b[i-1]*alpha[i-1]*np.sinh(b[i-1]*alpha[i-1]) - S[i-1]*np.sinh(b[i-1]*alpha[i-1])*deprime1[i-1];
        matrix[4*i + 1, 4*(i-1) + 1] = (S[i-1]/L[i-1])*b[i-1]*alpha[i-1]*np.cosh(b[i-1]*alpha[i-1]) - S[i-1]*np.cosh(b[i-1]*alpha[i-1])*deprime2[i-1];
        matrix[4*i + 1, 4*(i-1) + 2] = -(S[i-1]/L[i-1])*b[i-1]*beta[i-1]*np.sin(b[i-1]*beta[i-1]) - S[i-1]*np.sin(b[i-1]*beta[i-1])*deprime3[i-1];
        matrix[4*i + 1, 4*(i-1) + 3] = (S[i-1]/L[i-1])*b[i-1]*beta[i-1]*np.cos(b[i-1]*beta[i-1]) - S[i-1]*np.cos(b[i-1]*beta[i-1])*deprime4[i-1];
        matrix[4*i + 1, 4*(i-1) + 5] = -(S[i]/L[i])*b[i]*alpha[i] + S[i]*deprime2[i];
        matrix[4*i + 1, 4*(i-1) + 7] = -(S[i]/L[i])*b[i]*beta[i] + S[i]*deprime4[i];
    
    #Right boundary conditions, 2 equations
    matrix[4*N - 2, 4*N - 4] = b[N-1]*alpha[N-1]*np.cosh(b[N-1]*alpha[N-1])*deprime1[N-1];
    matrix[4*N - 2, 4*N - 3] = b[N-1]*alpha[N-1]*np.sinh(b[N-1]*alpha[N-1])*deprime2[N-1];
    matrix[4*N - 2, 4*N - 2] = b[N-1]*beta[N-1]*np.cos(b[N-1]*beta[N-1])*deprime3[N-1];
    matrix[4*N - 2, 4*N - 1] = -b[N-1]*beta[N-1]*np.sin(b[N-1]*beta[N-1])*deprime4[N-1];
    
    matrix[4*N - 1, 4*N - 4] = b[N-1]*alpha[N-1]*np.sinh(b[N-1]*alpha[N-1]) - L[N-1]*np.sinh(b[N-1]*alpha[N-1])*deprime1[N-1];
    matrix[4*N - 1, 4*N - 3] = b[N-1]*alpha[N-1]*np.cosh(b[N-1]*alpha[N-1]) - L[N-1]*np.cosh(b[N-1]*alpha[N-1])*deprime2[N-1];
    matrix[4*N - 1, 4*N - 2] = -b[N-1]*beta[N-1]*np.sin(b[N-1]*beta[N-1]) - L[N-1]*np.sin(b[N-1]*beta[N-1])*deprime3[N-1];
    matrix[4*N - 1, 4*N - 1] = b[N-1]*beta[N-1]*np.cos(b[N-1]*beta[N-1]) - L[N-1]*np.cos(b[N-1]*beta[N-1])*deprime4[N-1];
    
    return matrix


def beammodefreq(rho, k, E, G, w, h, L, f0, epsilon, n):
    
    # This function scans over a range of frequencies to determine where the 
    # determinant of our mode matrix is 0, which correspond to our desired mode
    # frequencies.
    
    # f0 = starting point of frequency scan
    # epsilon = mode frequency resolution
    # n = number of mode frequencies to find

    modefreq = np.zeros(n);
    
    df0 = 100;
    df = df0;
    f = f0; 
     
    for i in range(n):
        
        while df > epsilon:
            
            det1 = np.linalg.det(modematrix(rho, k, E, G, w, h, L, 2*np.pi*f));
            det2 = np.linalg.det(modematrix(rho, k, E, G, w, h, L, 2*np.pi*(f + df)));
            
            if det1*det2 < 0:
                
                df = 0.1*df;
                
            else:
                
                f = f + df;
            
        modefreq[i] = f;
    
        df = df0;
    
        f = f + df;


    return modefreq


def modecoefficients(rho, k, E, G, w, h, L, modefreq):
    
    # Computes coefficients of whichever mode has fequency modefreq, as described
    # in the 3rd paragraph of section C of B. H. Suits 2000.
    
    omega = 2*np.pi*modefreq; #nth mode angular frequency
    
    M = modematrix(rho, k, E, G, w, h, L, omega); #Get modematrix at omega0
    
    Mred = M[1:, 1:]; #Delete first row and column of modematrix
    
    b = -M[1:,0]; #Get all rows except the first from the first column of modematrix, shift to rhs of equation.
    
    C = np.linalg.solve(Mred,b); #Determine coefficients C2 to C7, having already set C1=1.
    
    C = np.concatenate(([1.0],C)); #Add C1 = 1 to complete vector of coefficients.
    
    return C


def modedisplacement(rho, k, E, G, w, h, L, modefreq, x):
    
    # This function computes the displacement of one of the modes of our beam at 
    # point x along its length. It's used to plot the shape of a given mode.
    
    C = modecoefficients(rho, k, E, G, w, h, L, modefreq);

    #Compute a whole helluvalot of constants
    S = w * h; #Beam cross-sectional areas
    
    omega = 2*np.pi*modefreq;
    
    I = w * h**3 / 12; #Cross-section area moments of inertia    
    
    b = np.sqrt(rho * S * L**4 * omega**2 / (E * I));
    r = np.sqrt(I / (S * L**2));
    s = np.sqrt((E / (k * G)) * r**2);
    
    #Correct alpha and beta definitions from Huang '61
    alpha = np.sqrt(1/2) * np.sqrt(-(r**2 + s**2) + np.sqrt((r**2 - s**2)**2 + 4/b**2));
    beta = np.sqrt(1/2) * np.sqrt(r**2 + s**2 + np.sqrt((r**2 - s**2)**2 + 4/b**2));
    
    #Find which segment x lies in
    
    interfacexvalues = np.cumsum(L); # Get x positions of the interfaces between segments
    interfacexvaluespluszero = np.concatenate(([0], interfacexvalues));
    
    k = np.digitize(x,interfacexvalues);
        
    # x lies within the kth segment
        
    xi = (x - interfacexvaluespluszero[k]) / L[k]; #Compute normalized x position xi within kth segment
        
    Y = C[4*k]*np.cosh(b[k]*alpha[k]*xi) + C[4*k + 1]*np.sinh(b[k]*alpha[k]*xi) + C[4*k + 2]*np.cos(b[k]*beta[k]*xi) + C[4*k + 3]*np.sin(b[k]*beta[k]*xi);

    return Y
    

def modeplotter(rho, k, E, G, w, h, L, modefreq):
    
    # This function computes the shape of a desired mode and plots it.
    
    step = 0.001;
    
    #Color list
    color = ['black', 'r'];
    
    interfacexvalues = np.cumsum(L); # Get x positions of the interfaces between segments
    interfacexvaluespluszero = np.concatenate(([0], interfacexvalues));
    
    for i in range(len(h)):
        
        x = np.arange(interfacexvaluespluszero[i], interfacexvaluespluszero[i+1]-step, step);
        
        Y = np.zeros(len(x));
        
        for j in range(len(x)):
            
            Y[j] = modedisplacement(rho, k, E, G, w, h, L, modefreq, x[j]);
            
        plt.plot(x, Y, color[np.mod(i,2)], linewidth = np.floor(300*h[i])); 
        
    plt.show()
    
    
def nodefinder(rho, k, E, G, w, h, L, modefreq, Nmax):
    
    # Find the first node of a given mode, via bisection method.
    # Assumes it's the fundamental, as that's the only mode whose node I care about
    
    totalL = np.sum(L); #Find total length of beam
    
    #Initial guess, it's in the left half of the beam.
    a = 0;
    b = totalL/2;
    
    for i in range(Nmax):
        
        c = (b + a)/2;
        
        fa = modedisplacement(rho, k, E, G, w, h, L, modefreq, a);

        fc = modedisplacement(rho, k, E, G, w, h, L, modefreq, c);

        if np.sign(fa) == np.sign(fc):
            
            a = c;
            
        else: 
            
            b = c;
            
    return c
        
    
    
    
    
    
    
    
    
    
    
    
    











