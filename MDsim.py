#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  6 11:55:59 2022

@author: hirshb
"""

#imports
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as BOLTZMANN

    ################################################################
    ################## NO EDITING BELOW THIS LINE ##################
    ################################################################

class Simulation:
    
    def __init__( self, dt, L, Nsteps=0, R=None, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, \
                 outname="sim.log", debug=False ):
        
        #general        
        self.debug=debug
        self.printfreq = printfreq
        self.xyzfile = open( xyzname, 'w' )
        self.outfile = open( outname, 'w' )
        
        #simulation
        self.Nsteps = Nsteps
        self.dt = dt
        self.L = L
        self.seed = seed
        self.step = step
        self.forcetype = "eval" + ftype
        self.fac = fac #factor for printing of XYZ in some units other than m
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        else:
            self.R = np.zeros( (1,3) )
            self.mass = 1.6735575E-27 #H mass in Kg
            self.kind = ["H"]
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
            
        if p is not None:
            self.p = p
            self.K = K
        else:
            self.p = np.zeros( (self.Natoms,3) )
            self.K = 0.0
        
        if F is not None:
            self.F = F
            self.U = U
        else:
            self.F = np.zeros( (self.Natoms,3) )
            self.U = 0.0
               
        #set RNG seed
        np.random.seed( self.seed)
    
    def __del__( self ):
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FORCETYPE AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).

        Returns
        -------
        None. Calls the correct method based on self.forcetype.

        """
        
        getattr(self, self.forcetype)(**kwargs)
            
    def dumpThermo( self ):
        """
        THIS FUNCTION DUMPS THE ENERGY OF THE SYSTEM TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
        if( self.step == 0 ):
            self.outfile.write( "step K U E \n" )
        
        self.outfile.write( str(self.step) + " " \
                          + "{:.6e}".format(self.K) + " " \
                          + "{:.6e}".format(self.U) + " " \
                          + "{:.6e}".format(self.E) + "\n" )
                
    def dumpXYZ( self ):
        """
        THIS FUNCTION DUMP THE COORDINATES OF THE SYSTEM IN XYZ FORMAT TO FILE.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
            
        self.xyzfile.write( str( self.Natoms ) + "\n")
        self.xyzfile.write( "Step " + str( self.step ) + "\n" )
        
        for i in range( self.Natoms ):
            self.xyzfile.write( self.kind[i] + " " + \
                               "{:.6e}".format( self.R[i,0]*self.fac ) + " " + \
                               "{:.6e}".format( self.R[i,1]*self.fac ) + " " + \
                               "{:.6e}".format( self.R[i,2]*self.fac ) + "\n" )
    
    def readXYZ( self, inpname ):
        """
        THIS FUNCTION READS THE COORDINATES IN XYZ FORMAT. USE ONE STEP.

        Parameters
        ----------
        R : numpy.ndarray
            Natoms x 3 array with particle coordinates.
        step : int
            Simulation time step.
        file : _io.TextIOWrapper 
            File object to write to. Created using open() in main.py

        Returns
        -------
        None.

        """
           
        df = pd.read_csv( inpname, sep="\s+", skiprows=2, header=None )
        
        self.kind = df[ 0 ]
        self.R = df[ [1,2,3] ].to_numpy()
        self.Natoms = self.R.shape[0]
        
    ################################################################
    ################## NO EDITING ABOVE THIS LINE ##################
    ################################################################
    
    
    def sampleMB( self, temp, removeCM=True ):
        """
        THIS FUNCTIONS SAMPLES INITIAL MOMENTA FROM THE MB DISTRIBUTION.
        IT ALSO REMOVES THE COM VELOCITY, IF REQUESTED.

        Parameters
        ----------
        temp : float
            The temperature to sample from.
        removeCM : bool, optional
            Remove COM velocity or not. The default is True.

        Returns
        -------
        None. Sets the value of self.p.

        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
    
    def applyPBC( self ):
        """
        THIS FUNCTION APPLIES PERIODIC BOUNDARY CONDITIONS.

        Returns
        -------
        None. Sets the value of self.R.

        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
                    
    def removeRCM( self ):
        """
        THIS FUNCTION REMOVES THE CENTERS OF MASS MOTION

        Returns
        -------
        None. Sets the value of self.R.

        """    
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################        
             
    def evalLJ( self, eps, sig ):
        """
        THIS FUNCTION EVALUTES THE LENNARD-JONES POTENTIAL AND FORCE.

        Parameters
        ----------
        eps : float
            epsilon LJ parameter.
        sig : float
            sigma LJ parameter.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """
        
        if( self.debug ):
            print( "Called evalLJ with eps = " \
                  + str(eps) + ", sig= " + str(sig)  )
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
            
    def evalHarm( self, omega ):
        """
        THIS FUNCTION EVALUTES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.

        Parameters
        ----------
        omega : float
            The frequency of the trap.

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """
        
        if( self.debug ):
            print( "Called evalHarm with omega = " + str(omega) )
            
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################

    def evalAnharm( self, Lambda ):
        """
        THIS FUNCTION EVALUTES THE POTENTIAL AND FORCE FOR AN ANHARMONIC TRAP.

        Parameters
        ----------
        Lambda : float
            The parameter of the trap U = 0.25 * Lambda * x**4

        Returns
        -------
        None. Sets the value of self.F and self.U.

        """
    
        if( self.debug ):
            print( "Called evalAnharm with Lambda = " + str(Lambda) ) 
            
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        
    def CalcKinE( self ):
        """
        THIS FUNCTIONS EVALUATES THE KINETIC ENERGY OF THE SYSTEM.

        Returns
        -------
        None. Sets the value of self.K.

        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        
    def VVstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE VELOCITY VERLET STEP.

        Returns
        -------
        None. Sets self.R, self.p.
        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        
    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES A SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO EVALUATE THE FORCES, PROPAGATE
        FOR NS TIME STEPS USING THE VELOCITY VERLET ALGORITHM, APPLY PBC, 
        AND CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME 
        STEP. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES.

        Returns
        -------
        None.

        """      
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################  