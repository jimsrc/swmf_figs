#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import os, sys, argparse, gc
import h5py
import numpy as np
from mpi4py import MPI
from glob import glob
# shared libs
sys.path.insert(0, os.environ['HOME']+'/my_projects/swmf_figs/src')
import shared.funcs as sf

#--- retrieve args
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
'-ds', '--dir_src',
type=str,
default=None,
help='input ASCII file',
)
parser.add_argument(
'-dd', '--dir_dst',
type=str,
default='<same-as-DIR_SRC>',
help='output hdf5 file'
)
parser.add_argument(
'-resume', '--resume',
action='store_true',
default=False,
help='Use it if you don\'t want to override the .h5 files in destination dir.',
)
parser.add_argument(
'-ndo', '--ndo',
type=int,
default=0,
help="""
Max number of files to process **per processor**.
If 0, it process all the input files it finds in DIR_SRC directory.
"""
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
#--- defaults
if pa.dir_dst=='<same-as-DIR_SRC>': 
    pa.dir_dst=pa.dir_src
else:
    os.system('mkdir -p '+pa.dir_dst)


# _data is of shape (4,:) where ':' is the size of rows in the ASCII file
vnames = custom_data.vnames # primitive columns in ASCII file


#---------------------------- MPI
comm    = MPI.COMM_WORLD
rank    = comm.Get_rank() # proc rank
wsize   = comm.Get_size() # nmbr of procs

#--- distribute the work
# complete list of input files
fnm_list      = glob(pa.dir_src+'/*.out')
# list of number of files assigned to each proc
nf_proc       = sf.equi_list(fnm_list, wsize)
# nmbr of files assigned to all previous procs (i.e. those ranks < 'rank')
nf_prev       = nf_proc[:rank].sum() 
# list of files assigned to this proc
fnm_list_proc = fnm_list[nf_prev:nf_prev+nf_proc[rank]]

ndone = 0 # number of processed files per processor.
for fname_inp in fnm_list_proc:
    print " [rank:%d] processing: %s" % (rank, fname_inp.split('/')[-1])
    # check is output already exists
    fname_out = pa.dir_dst +'/'+ fname_inp.split('/')[-1].replace('.out','.h5')
    if pa.resume and os.path.isfile(fname_out):
        continue # dont't override .h5 file

    _data  = sf.read_data(fname_inp, vnames)

    # we have to manually garbage-collect this. See:
    # https://sourceforge.net/p/spacepy/bugs/99/
    gc.collect()

    vdict = {}

    # list of variable names (associated to some of the process_..() functions
    # in the custom_data.py file) that will be saved into the HDF5 output file.
    retrieve_vnames = custom_data.ovnames
    for rvname in retrieve_vnames:
        print " [*] vname: " + rvname +'\n'
        # we'll obtain:
        # - same data but in structured way; friendly for plot_surface().
        # - fill 'vdict' with original ASCII data
        vdict[rvname] = sf.get_array_vars(data=_data, checks=False, 
            complete_domain_walk=True, 
            vnames=custom_data.vnames, 
            data_processor=getattr(custom_data, 'process_'+rvname),
            vectorial=False if rvname not in custom_data.ovectors else True,
            )
    print "[*] multi-dimensional data built ok."

    # NOTE: d['data'] is processed sutff built from the original (from
    # the ASCII file) simulation data. Such processing was made
    # by 'data_processor()'.
    # NOTE: a this point, 'vdict' has the original data from the ASCII file.
    r, ph, th   = vdict[rvname]['coords']

    fo = h5py.File(fname_out, 'w')
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
    print " [*] saving: " + fo.filename
    fo.close()
    del fo, vdict, _data
    ndone += 1
    if (pa.ndo > 0) and (ndone > pa.ndo): break


print "\n[r:%d] Finished!" % (rank)

#EOF
