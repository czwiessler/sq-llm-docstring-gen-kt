from mpi4py import MPI
import numpy as np
import sys,os
import camb
import time
from pyDOE import *

###### MPI DEFINITIONS ######                                    
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

#################################### INPUT #############################################
dimensions = 7
points     = 2000

# CAMB parameters
hierarchy    = 'degenerate'
Nnu          = 3   #number of massive neutrinos
Neff         = 3.046
tau          = 0.0925   # REPS takes 0.0925
Omega_k      = 0.0
pivot_scalar = 0.05       # hard coded into REPS
pivot_tensor = 0.05       # hard coded into REPS
kmax         = 100.0
k_per_logint = 30
redshifts    = [0]
Tcmb         = 2.7255     # hard coded into REPS

#              [Om,  Ob,   h,   ns,  s8,  Mnu,  w]
Min = np.array([0.1, 0.03, 0.5, 0.8, 0.6, 0.0, -1.3])
Max = np.array([0.5, 0.07, 0.9, 1.2, 1.0, 1.0, -0.7])

# whether generate standard or paired fixed simulations
standard = True
########################################################################################

# get latin hypercube
np.random.seed(6)
coords = lhs(dimensions, samples=points, criterion='c')

# check if the generated parameters are the same as the saved ones
f_params = 'latin_hypercube_params.txt'
if os.path.exists(f_params):  
    params  = np.loadtxt(f_params)
    params2 = Min + coords*(Max-Min)
    rdiff = np.absolute((params - params2)/params2)
    if np.max(rdiff)>1e-8:
        raise Exception('Cosmological parameters are different!!!')
else:
    np.savetxt(f_params, Min+coords*(Max-Min), fmt='%.5f')


# get the numbers each cpu will work on
numbers = np.where(np.arange(points)%nprocs==myrank)[0]


# do a loop over all the points in the latin hypercube
for i in numbers:

    # create output folder in case it does not exists
    folder = '/mnt/ceph/users/fvillaescusa/Quijote/nwLH/%s'%i
    os.makedirs(folder, exist_ok=True)

    # create initial conditions folder if it does not exists
    folder_ICs = '%s/ICs'%folder
    os.makedirs(folder_ICs, exist_ok=True)
    folder_CAMB = '%s/ICs/CAMB'%folder
    os.makedirs(folder_CAMB, exist_ok=True)
    folder_CLASS = '%s/ICs/CLASS'%folder
    os.makedirs(folder_CLASS, exist_ok=True)

    # find the values of the cosmological parameters
    coord = Min + coords[i]*(Max-Min)

    Omega_m = coord[0]
    Omega_b = coord[1]
    h       = coord[2]
    ns      = coord[3]
    s8      = coord[4]
    Mnu     = coord[5]
    w       = coord[6]

    g = open('%s/Cosmo_params.dat'%folder, 'w')
    g.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f\n'%(Omega_m, Omega_b, h, ns, s8,Mnu,w))
    g.close()

    print('realization %d'%i)
    print('Omega_m = %.5f'%Omega_m)
    print('Omega_b = %.5f'%Omega_b)
    print('h       = %.5f'%h)
    print('ns      = %.5f'%ns)
    print('s8      = %.5f'%s8)
    print('Mnu     = %.5f'%Mnu)
    print('w       = %.5f'%w)
    
    ##### compute As in order to match s8 using CAMB #####
    
    Omega_c = Omega_m - Omega_b - Mnu/(93.14*h**2)
    Omega_n = Mnu/(93.14*h**2)

    #get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100, ombh2=Omega_b*h**2, omch2=Omega_c*h**2, mnu=Mnu, TCMB=Tcmb, omk=Omega_k, tau=tau,
                       nnu=Neff, standard_neutrino_neff=Neff, num_massive_neutrinos=Nnu, neutrino_hierarchy=hierarchy)
    pars.set_dark_energy(w=w, dark_energy_model='fluid')
    
    pars.InitPower.set_params(As=2e-9 , ns=ns,
                             pivot_scalar=pivot_scalar, pivot_tensor=pivot_tensor)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0.], kmax=kmax, k_per_logint=k_per_logint)

    # compute spectra to find sigma8 for rescaling
    pars.set_accuracy(AccuracyBoost=4, lAccuracyBoost=4, lSampleBoost=4, HighAccuracyDefault=True, DoLateRadTruncation=True)#, high_accuracy_default=1)#, neutrino_q_boost=4)
    #pars.set_matter_power(accurate_massive_neutrino_transfers=1, nonlinear=True)
    results = camb.get_results(pars)
    s8_0 = results.get_sigma8()

    # compute As
    As = (2e-9 * (s8 / s8_0)**2)[0]
    print(As, s8, s8_0)
    
    
    ##### create REPS parameter file ####
    
    r_file="""
    # Boltzmann code
    boltzmann_code     =  camb
    boltzmann_folder   =  /mnt/home/fvillaescusa/software/CAMB-Jan2017/
    k_per_logint_camb  =  %d

    # input/output options
    neutrino_tabulated_functions = /mnt/home/fvillaescusa/software/reps/tabulated_functions/FF_GG_func_tab.dat
    #input_file         =  boundary_conditions.dat
    use_boundary_conditions_from_file = F
    #boundary_conditions_file = boundary_conditions.dat
    which_bc           =  0
    outputfile         =  nwLH
    output_format      =  ngenic transfer function
    print_hubble       =  T
    compute_Pk_0       =  T

    # cosmological parameters
    h                  =  %.5f
    OB0                =  %.5e
    OC0                =  %.5e
    OG0                =  2.469e-05
    w0                 =  %.5f
    wa                 =  0.0
    As                 =  %.5e
    tau_reio           =  %.5f
    ns                 =  %.5f
    kmax               =  %.5f

    # neutrino parameters
    Neff               =  0.00641
    M_nu               =  %.5f
    N_nu               =  3.0
    wrong_nu           =  1

    # redshifts
    z_final            =  0.
    z_initial          = 127.
    output_number      =  2
    z_output           =  0.0 127.0
    """%(k_per_logint, h, Omega_b, Omega_c, w, As, tau, ns, kmax, Mnu)
    
    # save parameters to file
    f = open('%s/REPS.param'%folder_CAMB, 'w')
    f.write(r_file)
    f.close()

    os.chdir('%s'%folder_CAMB)
    try:   os.system('reps REPS.param')
    except:  pass
        
    #os.system('wait 10s')    


    r_file="""
    # Boltzmann code
    boltzmann_code     =  class
    boltzmann_folder   =  /mnt/home/fvillaescusa/software/class_public/
    k_per_logint_camb  =  %d

    # input/output options
    neutrino_tabulated_functions = /mnt/home/fvillaescusa/software/reps/tabulated_functions/FF_GG_func_tab.dat
    #input_file         =  boundary_conditions.dat
    use_boundary_conditions_from_file = F
    #boundary_conditions_file = boundary_conditions.dat
    which_bc           =  0
    outputfile         =  nwLH
    output_format      =  ngenic transfer function
    print_hubble       =  T
    compute_Pk_0       =  T

    # cosmological parameters
    h                  =  %.5f
    OB0                =  %.5e
    OC0                =  %.5e
    OG0                =  2.469e-05
    w0                 =  %.5f
    wa                 =  0.0
    As                 =  %.5e
    tau_reio           =  %.5f
    ns                 =  %.5f
    kmax               =  %.5f

    # neutrino parameters
    Neff               =  0.00641
    M_nu               =  %.5f
    N_nu               =  3.0
    wrong_nu           =  1

    # redshifts
    z_final            =  0.
    z_initial          = 127.
    output_number      =  2
    z_output           =  0.0 127.0
    """%(k_per_logint, h, Omega_b, Omega_c, w, As, tau, ns, kmax, Mnu)
    
    # save parameters to file
    f = open('%s/REPS.param'%folder_CLASS, 'w')
    f.write(r_file)
    f.close()

    os.chdir('%s'%folder_CLASS)
    #os.system('reps REPS.param')
    try:   
        stream = os.popen('reps REPS.param')
        output = stream.read()
        print(output)
    except:  
        print('Error!!!')
        pass

    ########################## create Hz file ###################################  
    f_in  = 'nwLH_hubble.txt'   #file provided by Matteo  
    f_out = '%s/Hz.txt'%folder  #file to be used by Gadget  
    bins  = 1000                #number of bins in the new file  
    z_max = 127.0               #maximum redshift of the new file  
    z_min = 0.0                 #minimum redshift of the new file  

    #read original file; H(z) in km/s/Mpc
    while not(os.path.exists(f_in)):
        time.sleep(10)
    z,Hz = np.loadtxt(f_in,unpack=True)

    #create a new (1+z) array    
    z_new = np.logspace(np.log10(1.0+z_min),np.log10(1.0+z_max),bins)

    #interpolate from the original table. The H(z) has to be in Gadget units:  
    #km/s/(kpc/h), thus we need to multiply the Hz of Matteo by 1.0/(1000.0*h)    
    Hz_new = np.interp(z_new-1,z,Hz)/(Hz[0]/100.0)/1000.0

    #create the w=-1 array     
    w_dumb = np.ones(bins)*(-1.0)

    #the format of the H(z) file is 1+z,w,H(z), where 1+z should be decreasing   
    np.savetxt(f_out,np.transpose([z_new[::-1], w_dumb, Hz_new[::-1]]))
    ############################################################################# 

    # get the value of Hzi
    Hzi = Hz_new[-1]
    print('Hzi = %.4f'%Hzi)

    ########################## write N-GenIC parameter file #######################
    # parameter file for standard simulations
    a="""
    Nmesh               1024      
    Nsample             512                
    Box                 1000000.0   
    FileBase            ics 
    OutputDir           ./  
    GlassFile           /mnt/ceph/users/fvillaescusa/Quijote/nwLH/Codes/n-genic_growth/GLASS/dummy_glass_NU_64_64.dat 
    GlassTileFac        8    
    Omega               %.5f    
    OmegaLambda         %.5f    
    OmegaBaryon         0.0000    
    OmegaDM_2ndSpecies  %.5e    
    HubbleParam         %.5f    
    Redshift            127       
    Sigma8              %.5f       
    Hzi                 %.4f   
    SphereMode       0                                    		         
    WhichSpectrum    2 
    FileWithInputSpectrum   ./nwLH_Pm_rescaled_z127.0000.txt 
    FileWithTransfer        ./nwLH_rescaled_transfer_z127.0000.txt
    FileWithBGrowth         ./nwLH_fcb_z127.0000.txt
    FileWithCDMGrowth       ./nwLH_fcb_z127.0000.txt
    FileWithNUGrowth        ./nwLH_fn_z127.0000.txt
    InputSpectrum_UnitLength_in_cm  3.085678e24 
    ReNormalizeInputSpectrum        0               
    ShapeGamma                      0.201     
    PrimordialIndex                 1.0       
    Phase_flip                      0         
    RayleighSampling                1         
    Seed                            %d        
    NumFilesWrittenInParallel       8  
    UnitLength_in_cm                3.085678e21  
    UnitMass_in_g                   1.989e43     
    UnitVelocity_in_cm_per_s        1e5          
    WDM_On                          0      
    WDM_Vtherm_On                   0      
    WDM_PartMass_in_kev             10.0   
    NU_On                           1
    NU_Vtherm_On                    1
    NU_PartMass_in_ev               %.5e
    """%(Omega_m, 1.0-Omega_m, Omega_n, h, s8, Hzi, i+6700, Mnu)
    
    # save parameters to file
    f = open('%s/NGenIC.param'%folder_ICs, 'w');  f.write(a);  f.close()
    #############################################################################

    ####################### write Gadget3 parameter file ########################
    a="""InitCondFile              ./ICs/ics
OutputDir                 ./
OutputListFilename        /mnt/ceph/users/fvillaescusa/Quijote/nwLH/times.txt
NumFilesPerSnapshot       8
NumFilesWrittenInParallel 8
CpuTimeBetRestartFile     10800.0   
TimeLimitCPU              10000000  
ICFormat                  1
SnapFormat                3
TimeBegin                 0.0078125     
TimeMax	                  1.00          
Omega0	                  %.4f    
OmegaLambda               %.4f
OmegaBaryon               0.0000     
HubbleParam               %.4f     
BoxSize                   1000000.0
DarkEnergyFile            ./ICs/Hz.txt
Time_tree_on_nu           0.09  

SofteningGas              0.0
SofteningHalo             50.0   
SofteningDisk             50.0
SofteningBulge            0.0
SofteningStars            0.0
SofteningBndry            0.0
SofteningGasMaxPhys       0.0
SofteningHaloMaxPhys      50.0
SofteningDiskMaxPhys      50.0
SofteningBulgeMaxPhys     0.0
SofteningStarsMaxPhys     0.0
SofteningBndryMaxPhys     0.0

PartAllocFactor           2.5  
MaxMemSize	              14500
BufferSize                120
CoolingOn                 0
StarformationOn           0

TypeOfTimestepCriterion   0   	                    
ErrTolIntAccuracy         0.006  
MaxSizeTimestep           0.025
MinSizeTimestep           0.0

ErrTolTheta               0.5
TypeOfOpeningCriterion    1
ErrTolForceAcc            0.001
TreeDomainUpdateFrequency 0.01
 
DesNumNgb                 33
MaxNumNgbDeviation        2
ArtBulkViscConst          1.0
InitGasTemp               273.0  
MinGasTemp                10.0    
CourantFac                0.15
ComovingIntegrationOn     1    
PeriodicBoundariesOn      1    
MinGasHsmlFractional      0.1  
OutputListOn              1    
TimeBetSnapshot           1.   
TimeOfFirstSnapshot       1.   
TimeBetStatistics         0.5  
MaxRMSDisplacementFac     0.25 
EnergyFile                energy.txt
InfoFile                  info.txt
TimingsFile               timings.txt
CpuFile                   cpu.txt
TimebinFile               Timebin.txt
SnapshotFileBase          snap
RestartFile               restart
ResubmitOn                0
ResubmitCommand           /home/vspringe/autosubmit
UnitLength_in_cm          3.085678e21      
UnitMass_in_g             1.989e43         
UnitVelocity_in_cm_per_s  1e5              
GravityConstantInternal   0
    """%(Omega_m, 1.0-Omega_m, h)

    f = open('%s/G3.param'%folder, 'w');  f.write(a);  f.close()
    #############################################################################
