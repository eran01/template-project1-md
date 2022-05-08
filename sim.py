#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  6 11:55:59 2022

@author: hirshb


WELCOME TO YOUR FIRST PROJECT! THIS BIT OF TEXT IS CALLED A DOCSTRING.
BELOW, I HAVE CREATED A CLASS CALLED "SIMULATION" FOR YOUR CONVENIENCE.
I HAVE ALSO IMPLEMENTED A CONSTRUCTOR, WHICH IS A METHOD THAT IS CALLED 
EVERY TIME YOU CREATE AN OBJECT OF THE CLASS USING, FOR EXAMPLE, 
    
    >>> mysim = Simulation( dt=0.1E-15, L=11.3E-10, ftype="LJ" )

I HAVE ALSO IMPLEMENTED SEVERAL USEFUL METHODS THAT YOU CAN CALL AND USE, 
BUT DO NOT EDIT THEM. THEY ARE: evalForce, dumpXYZ, dumpThermo and readXYZ.

YOU DO NOT NEED TO EDIT THE CLASS ITSELF. 

YOUR JOB IS TO IMPLEMENT THE LIST OF CLASS METHODS DEFINED BELOW WHERE YOU 
WILL SEE THE FOLLOWING TEXT: 

        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################

YOU ARE, HOWEVER, EXPECTED TO UNDERSTAND WHAT ARE THE MEMBERS OF THE CLASS
AND USE THEM IN YOUR IMPLEMENTATION. THEY ARE ALL EXPLAINED IN THE 
DOCSTRING OF THE CONSTRUCTOR BELOW. FOR EXAMPLE, WHENEVER YOU WISH 
TO USE/UPDATE THE MOMENTA OF THE PARTICLES IN ONE OF YOUR METHODS, YOU CAN
ACCESS IT BY USING self.p. 

    >>> self.p = np.zeros( (self.Natoms,3) )
        
FINALLY, YOU WILL NEED TO EDIT THE run.py FILE WHICH RUNS THE SIMULATION.
SEE MORE INSTRUCTIONS THERE.

"""
################################################################
################## NO EDITING BELOW THIS LINE ##################
################################################################

#imports
import numpy as np
import pandas as pd
from scipy.constants import Boltzmann as BOLTZMANN
import matplotlib.pyplot as plt

class Simulation:
    
    def __init__( self, dt, L, Nsteps=0, R=None, mass=None, kind=None, \
                 p=None, F=None, U=None, K=None, seed=937142, ftype=None, \
                 step=0, printfreq=1000, xyzname="sim.xyz", fac=1.0, \
                 outname="sim.log", debug=False ):
        """
        THIS IS THE CONSTRUCTOR. SEE DETAILED DESCRIPTION OF DATA MEMBERS
        BELOW. THE DESCRIPTION OF EACH METHOD IS GIVEN IN ITS DOCSTRING.

        Parameters
        ----------
        dt : float
            Simulation time step.
            
        L : float
            Simulation box side length.
            
        Nsteps : int, optional
            Number of steps to take. The default is 0.
            
        R : numpy.ndarray, optional
            Particles' positions, Natoms x 3 array. The default is None.
            
        mass : numpy.ndarray, optional
            Particles' masses, Natoms x 1 array. The default is None.
            
        kind : list of str, optional
            Natoms x 1 list with atom type for printing. The default is None.
            
        p : numpy.ndarray, optional
            Particles' momenta, Natoms x 3 array. The default is None.
            
        F : numpy.ndarray, optional
            Particles' forces, Natoms x 3 array. The default is None.
            
        U : float, optional
            Potential energy . The default is None.
            
        K : float, optional
            Kinetic energy. The default is None.
        
        E : float, optional
            Total energy. The default is None.
                    
        seed : int, optional
            Big number for reproducible random numbers. The default is 937142.
            
        ftype : str, optional
            String to call the force evaluation method. The default is None.
            
        step : INT, optional
            Current simulation step. The default is 0.
            
        printfreq : int, optional
            PRINT EVERY printfreq TIME STEPS. The default is 1000.
            
        xyzname : TYPE, optional
            DESCRIPTION. The default is "sim.xyz".
            
        fac : float, optional
            Factor to multiply the positions for printing. The default is 1.0.
            
        outname : TYPE, optional
            DESCRIPTION. The default is "sim.log".
            
        debug : bool, optional
            Controls printing for debugging. The default is False.

        Returns
        -------
        None.

        """
        
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
        self.fac = fac 
        
        #system        
        if R is not None:
            self.R = R        
            self.mass = mass
            self.kind = kind
            self.Natoms = self.R.shape[0]
            self.mass_matrix = np.array( [self.mass,]*3 ).transpose()
        else:
            self.R = np.zeros( (1,3) )
            self.mass = 1.6735575E-27 #H mass in kg as default
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
            
        self.E = self.K + self.U
               
        #set RNG seed
        np.random.seed( self.seed )
        
        #check force type
        if ( ftype == "LJ" or ftype == "Harm" or ftype == "Anharm"):
            self.ftype = "eval" + ftype
        else:
            raise ValueError("Wrong ftype value - use LJ or Harm or Anharm.")
    
    def __del__( self ):
        """
        THIS IS THE DESCTRUCTOR. NOT USUALLY NEEDED IN PYTHON. 
        JUST HERE TO CLOSE THE FILES.

        Returns
        -------
        None.

        """
        self.xyzfile.close()
        self.outfile.close()
    
    def evalForce( self, **kwargs ):
        """
        THIS FUNCTION CALLS THE FORCE EVALUATION METHOD, BASED ON THE VALUE
        OF FTYPE, AND PASSES ALL OF THE ARGUMENTS (WHATEVER THEY ARE).

        Returns
        -------
        None. Calls the correct method based on self.ftype.

        """
        
        getattr(self, self.ftype)(**kwargs)
            
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
        THIS FUNCTION READS THE INITIAL COORDINATES IN XYZ FORMAT.

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
        IT ALSO REMOVES THE COM MOMENTA, IF REQUESTED.

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
        THIS FUNCTION ZEROES THE CENTERS OF MASS POSITION VECTOR.

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
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR A HARMONIC TRAP.

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
        THIS FUNCTION EVALUATES THE POTENTIAL AND FORCE FOR AN ANHARMONIC TRAP.

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
        
    def MCstep( self, **kwargs ):
        """
        THIS FUNCTIONS PERFORMS ONE METROPOLIS MC STEP IN THE NVT ENSEMBLE.
        YOU WILL NEED TO PROPOSE TRANSLATION MOVES, APPLY  
        PBC, CALCULATE THE CHANGE IN POTENTIAL ENERGY, ACCEPT OR REJECT, 
        AND CALCULATE THE ACCEPTANCE PROBABILITY. 

        Returns
        -------
        None. Sets self.R.
        """
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
    
    def runMC( self, **kwargs ):
        """ 
        THIS FUNCTION DEFINES AN MC SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO LOOP OVER MC STEPS, 
        PRINT THE COORDINATES AND ENERGIES EVERY PRINTFREQ TIME STEPS 
        TO THEIR RESPECTIVE FILES, SIMILARLY TO YOUR MD CODE.

        Returns
        -------
        None.

        """   
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################
        
    def run( self, **kwargs ):
        """
        THIS FUNCTION DEFINES A SIMULATION DOES, GIVEN AN INSTANCE OF 
        THE SIMULATION CLASS. YOU WILL NEED TO:
            1. EVALUATE THE FORCES (USE evaluateForce() AND PASS A DICTIONARY
                                    WITH ALL THE PARAMETERS).
            2. PROPAGATE FOR NS TIME STEPS USING THE VELOCITY VERLET ALGORITHM.
            3. APPLY PBC.
            4. CALCULATE THE KINETIC, POTENTIAL AND TOTAL ENERGY AT EACH TIME
            STEP. 
            5. YOU WILL ALSO NEED TO PRINT THE COORDINATES AND ENERGIES EVERY 
        PRINTFREQ TIME STEPS TO THEIR RESPECTIVE FILES, xyzfile AND outfile.

        Returns
        -------
        None.

        """      
        
        ################################################################
        ####################### YOUR CODE GOES HERE ####################
        ################################################################  