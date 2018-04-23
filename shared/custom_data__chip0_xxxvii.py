# These correspond to the fields of the 
# run in commit <FDIPS_iv>

#++++++++++++++++++++++++++++++++++++++++
# names of the fields in the ASCII file
# NOTE: this information can be checked from the *.info files in
# the output directory.
vnames = [
'bx', 'by', 'bz',
]

# variables (corresponding to the process_..() functions below) that
# will be saved into the HDF5 files. This is necessary when converting 
# from ASCII to HDF5.
ovnames = ['B',]

#++++++++++++++++++++++++++++++++++++++++
# methods that return scalars as function of 
# the original variables in the ASCII file
import numpy as np

def process_Bmod(vdict):
    try:
        bx, by, bz = vdict['bx'], vdict['by'], vdict['bz'] # when reading ASCII file
    except:
        bx, by, bz = vdict['B'].transpose((3,0,1,2)) # when reading the HDF5 file
    Bmod = np.sqrt(bx*bx + by*by + bz*bz)
    return Bmod

def process_B(vdict):
    bx, by, bz = vdict['bx'], vdict['by'], vdict['bz'] # when reading ASCII file
    return [bx,by,bz]

#def process_n(vdict):
#    return vdict['n']
#
#def process_temp(vdict):
#    return vdict['temp']
#
#def process_rho(vdict):
#    return vdict['rho']

#EOF
