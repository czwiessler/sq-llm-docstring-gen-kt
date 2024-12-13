# This script computes the matter Pk in real- and redshift-space. It takes as input
# the first and last number of the wanted realizations, the cosmology and the snapnum
# In redshift-space it computes the power spectrum along the 3 different axes. 
import argparse
from mpi4py import MPI
import numpy as np
import sys,os
import readgadget,readfof
import redshift_space_library as RSL
import Pk_library as PKL
import MAS_library as MASL

###### MPI DEFINITIONS ######                                    
comm   = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

# read the first and last realization to identify voids
parser = argparse.ArgumentParser(description="This script computes the bispectrum")
parser.add_argument("first",   help="first realization number", type=int)
parser.add_argument("last",    help="last  realization number", type=int)
parser.add_argument("cosmo",   help="folder with the realizations")
args = parser.parse_args()
first, last, cosmo = args.first, args.last, args.cosmo


# This routine computes the Pk in real- and redshift-space
def compute_Pk(snapshot, grid, MAS, threads, NCV, pair, 
               folder_out, cosmo, i, suffix, z, ptype):

    if NCV:  #paired-fixed simulations

        # real-space
        fpk = '%s/%s/NCV_%d_%d/Pk_%s_z=%s.txt'%(folder_out,cosmo,pair,i,suffix,z)
        if not(os.path.exists(fpk)):
            do_RSD, axis, save_multipoles = False, 0, False
            find_Pk(snapshot, grid, MAS, do_RSD, axis, threads, ptype,
                    fpk, save_multipoles)

    else:  #standard simulations

        # real-space
        fpk = '%s/%s/%d/Pk_%s_z=%s.txt'%(folder_out,cosmo,i,suffix,z)
        if not(os.path.exists(fpk)):
            do_RSD, axis, save_multipoles = False, 0, False
            find_Pk(snapshot, grid, MAS, do_RSD, axis, threads, ptype,
                    fpk, save_multipoles)



# This routine computes and saves the power spectrum
def find_Pk(snapshot, grid, MAS, do_RSD, axis, threads, ptype,
            fpk, save_multipoles):

    if os.path.exists(fpk):  return 0
    
    # read header
    head     = readgadget.header(snapshot)
    BoxSize  = head.boxsize/1e3  #Mpc/h  
    Nall     = head.nall         #Total number of particles
    Masses   = head.massarr*1e10 #Masses of the particles in Msun/h                    
    Omega_m  = head.omega_m
    Omega_l  = head.omega_l
    redshift = head.redshift
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#km/s/(Mpc/h)
    h        = head.hubble

    # read snapshot
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #Mpc/h

    # move particles to redshift-space
    if do_RSD:
        vel = readgadget.read_block(snapshot, "VEL ", ptype) #km/s
        RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

    # calculate Pk
    delta = np.zeros((grid,grid,grid), dtype=np.float32)
    if len(ptype)>1:  #for multiple particles read masses
        mass = np.zeros(pos.shape[0], dtype=np.float32)
        offset = 0
        for j in ptype:
            mass[offset: offset+Nall[j]] = Masses[j]
            offset += Nall[j]
        MASL.MA(pos, delta, BoxSize, MAS, W=mass)
    else:
        MASL.MA(pos, delta, BoxSize, MAS)
    delta /= np.mean(delta, dtype=np.float64);  delta -= 1.0 
    Pk = PKL.Pk(delta, BoxSize, axis, MAS, threads)

    # save results to file
    if save_multipoles:
        np.savetxt(fpk, np.transpose([Pk.k3D, Pk.Pk[:,0], Pk.Pk[:,1], Pk.Pk[:,2]]),
                   delimiter='\t')
    else:
        np.savetxt(fpk, np.transpose([Pk.k3D, Pk.Pk[:,0]]), delimiter='\t')



##################################### INPUT #########################################
# folder containing the snapshots
root = '/simons/scratch/fvillaescusa/pdf_information'

# Pk parameters
grid    = 1024
MAS     = 'CIC'
threads = 2

# folder that containts the results
folder_out = '/simons/scratch/fvillaescusa/pdf_information/Pk/matter/'

z = 127
#####################################################################################


# create output folder if it does not exist
if myrank==0 and not(os.path.exists(folder_out+cosmo)):  
    os.system('mkdir %s/%s/'%(folder_out,cosmo))
comm.Barrier()

# get the realizations each cpu works on
numbers = np.where(np.arange(args.first, args.last)%nprocs==myrank)[0]
numbers = np.arange(args.first, args.last)[numbers]


######## standard simulations #########
for i in numbers:

    # find the snapshot
    snapshot = '%s/Snapshots/%s/%d/ICs/ics'%(root,cosmo,i)
    if not(os.path.exists(snapshot+'.0')) and not(os.path.exists(snapshot+'.0.hdf5')):
        continue

    # create output folder if it does not exists
    if not(os.path.exists('%s/%s/%d'%(folder_out,cosmo,i))):
        os.system('mkdir %s/%s/%d'%(folder_out,cosmo,i))

    # neutrinos are special
    if cosmo in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Mnu_p/', 'Mnu_pp/', 'Mnu_ppp/']:

        # compute CDM+Baryons Pk
        NCV, suffix, ptype, pair = False, 'cb', [1], 0
        compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                   folder_out, cosmo, i, suffix, z, ptype)

        # compute matter Pk
        NCV, suffix, ptype, pair = False, 'm', [1,2], 0
        compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                   folder_out, cosmo, i, suffix, z, ptype)

    else:
        # compute matter Pk
        NCV, suffix, ptype, pair = False, 'm', [1], 0
        compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                   folder_out, cosmo, i, suffix, z, ptype)

    


###### paired fixed realizations ######
for i in numbers:

    for pair in [0,1]:
        snapshot = '%s/Snapshots/%s/NCV_%d_%d/ICs/ics'%(root,cosmo,pair,i)
        if not(os.path.exists(snapshot+'.0')) and not(os.path.exists(snapshot+'.0.hdf5')):
            continue

        print i,pair
        # create output folder if it does not exists
        if not(os.path.exists('%s/%s/NCV_%d_%d'%(folder_out,cosmo,pair,i))):
            os.system('mkdir %s/%s/NCV_%d_%d'%(folder_out,cosmo,pair,i))

        # neutrinos are special
        if cosmo in ['Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'Mnu_p/', 'Mnu_pp/', 'Mnu_ppp/']:

            # compute CDM+Baryons Pk
            NCV, suffix, ptype = True, 'cb', [1]
            compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                       folder_out, cosmo, i, suffix, z, ptype)

            # compute matter Pk
            NCV, suffix, ptype  = True, 'm', [1,2]
            compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                       folder_out, cosmo, i, suffix, z, ptype)

        else:
            # compute matter Pk
            NCV, suffix, ptype = True, 'm', [1]
            compute_Pk(snapshot, grid, MAS, threads, NCV, pair,
                       folder_out, cosmo, i, suffix, z, ptype)


        

