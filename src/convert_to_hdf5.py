#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import os, argparse
import h5py
import shared.funcs as sf
import numpy as np

#--- retrieve args
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
'-fi', '--fname_inp',
type=str,
default=None,
help='input ASCII file',
)
parser.add_argument(
'-fo', '--fname_out',
type=str,
default=None,
help='output hdf5 file'
)
# import custom methods for specific dataset
# NOTE: the names of the variables/observables depend on this 
# specific dataset && on the 'custom_data.py' script.
from shared import custom_data
data_methods = [nm for nm in dir(custom_data) if nm.startswith('process_')]
nmethods     = len(data_methods)
var_names    = [ dm.split('process_')[1] for dm in data_methods]
#parser.add_argument(
#'-vname', '--vname',
#type=str,
#help='scalar variable to plot. Available options: ' + ', '.join(var_names),
#default=var_names[0],
#)
pa = parser.parse_args()



# _data is of shape (4,:) where ':' is the size of rows in the ASCII file
vnames = custom_data.vnames # primitive columns in ASCII file
_data  = sf.read_data(pa.fname_inp, vnames)

vdict = {}

# list of variable names (associated to some of the process_..() functions
# in the custom_data.py file) that will be saved into the HDF5 output file.
retrieve_vnames = custom_data.ovnames
for rvname in retrieve_vnames:
    # we'll obtain:
    # - same data but in structured way; friendly for plot_surface().
    # - fill 'vdict' with original ASCII data
    vdict[rvname] = sf.get_array_vars(data=_data, checks=False, 
        complete_domain_walk=True, 
        vnames=custom_data.vnames, 
        data_processor=getattr(custom_data, 'process_'+rvname),
        vectorial=False if rvname not in ('B',) else True,
        )

# NOTE: d['data'] is processed sutff built from the original (from
# the ASCII file) simulation data. Such processing was made
# by 'data_processor()'.
# NOTE: a this point, 'vdict' has the original data from the ASCII file.
r, ph, th   = vdict[rvname]['coords']
#data        = d['data']; 


fo = h5py.File(pa.fname_out, 'w')
for rvname in retrieve_vnames:
    fo.create_dataset('data/'+rvname, dtype='f', 
        shape=vdict[rvname]['data'].shape, 
        fillvalue=np.nan, 
        compression='gzip', 
        compression_opts=9
        )
    fo['data/'+rvname][...] = vdict[rvname]['data']
    # NOTE: the shape of 'vdict[rvname]['data']' should be
    # either (r.size,ph.size,th.size) if scalar or
    # (r.size,ph.size,th.size,3) if vectorial variable.
        
fo['coords/r']  = r
fo['coords/ph'] = ph
fo['coords/th'] = th
fo.close()

#EOF