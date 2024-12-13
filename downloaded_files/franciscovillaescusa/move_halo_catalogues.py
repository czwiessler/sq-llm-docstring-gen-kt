# This script moves the halo catalogues from their original locations to the halo folder
import numpy as np
import sys,os

root = '/simons/scratch/fvillaescusa/pdf_information/'
##################################### INPUT ############################################
cosmologies = ['w_p', 'w_m']
#['Om_p', 'Ob_p', 'Ob2_p', 'h_p', 'ns_p', 's8_p',
#'Om_m', 'Ob_m', 'Ob2_m', 'h_m', 'ns_m', 's8_m',
#'Mnu_p', 'Mnu_pp', 'Mnu_ppp', 'fiducial', 'fiducial_LR','fiducial_HR',
#'latin_hypercube', 'DC_p', 'DC_m', 'w_p', 'w_m']
########################################################################################

# do a loop over all different cosmologies
for cosmo in cosmologies:

    # find the number of standard and paired fixed simulations
    paired_fixed_realizations = 0
    if   cosmo=='fiducial':         standard_realizations = 15000
    elif cosmo=='fiducial_ZA':      standard_realizations = 500
    elif cosmo=='fiducial_LR':      standard_realizations = 1000
    elif cosmo=='fiducial_HR':      standard_realizations = 100
    elif cosmo=='latin_hypercube':  standard_realizations = 2000
    else:                           standard_realizations = 500
     
    # find the name of the output halo folder containing all results
    folder = '%s/Halos/%s'%(root,cosmo)
    if not(os.path.exists(folder)):  os.system('mkdir %s'%folder)

    ###### standard realizations ######
    for i in range(standard_realizations):

        # create output folder if it does not exists
        folder_out = '%s/%d/'%(folder,i)
        if not(os.path.exists(folder_out)):  os.system('mkdir %s'%folder_out)

        # do a loop over the different redshifts
        for snapnum in [0,1,2,3,4]:

            folder_in = '%s/Snapshots/%s/%d/groups_%03d'%(root,cosmo,i,snapnum)
            if not(os.path.exists(folder_in)):  continue
            os.system('mv %s %s'%(folder_in, folder_out))


    ###### paired fixed realizations ######
    for i in range(paired_fixed_realizations):

        for pair in [0,1]:
            
            # create output folder if it does not exists
            folder_out = '%s/NCV_%d_%d/'%(folder,pair,i)
            if not(os.path.exists(folder_out)):  os.system('mkdir %s'%folder_out)

            # do a loop over the different redshifts
            for snapnum in [0,1,2,3,4]:

                folder_in = '%s/Snapshots/%s/NCV_%d_%d/groups_%03d'%(root,cosmo,pair,i,snapnum)
                if not(os.path.exists(folder_in)):  continue
                os.system('mv %s %s'%(folder_in, folder_out))



