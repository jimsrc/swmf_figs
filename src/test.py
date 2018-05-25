#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import argparse, os, glob, sys
import numpy as np
import shared.funcs as sf
from mpi4py import MPI
import fparsers as fp

#--- parse args
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="""
    This generates rates averaged over an interval of energy 
    channels (defined in 'ch_Eds').
    """,
)
# subparsers
subparsers = parser.add_subparsers(description=
    """
    Use one of the submodules below.
    """,
    )

# config the subparsers
for mod_name in ['3d_cut', 'r_cut']:
    # grab the class
    mod = getattr(fp, 'cutter__'+mod_name)()
    # grab the parser of class 'mod'
    #--- all subparsers will have this options in common:
    subparser_ = subparsers.add_parser(
    mod_name, 
    help = mod.help,
    parents = [mod.parser], 
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    subparser_.add_argument(
    '-fi', '--fname_inp',
    type=str,
    default=None,
    help='input ASCII file. If this option is used, don\'t use --dir_src.',
    )
    subparser_.add_argument(
    '-fig', '--fname_fig',
    type=str,
    default=None,
    help='output PNG file. If --dir_dst is specified, this option must not be used.',
    )
    subparser_.add_argument(
    '-ds', '--dir_src',
    type=str,
    default=None,
    help='input directory. If this option is used, don\'t use --fname_inp.',
    )
    subparser_.add_argument(
    '-dd', '--dir_dst',
    type=str,
    default=None,
    help='input directory',
    )
    # import custom methods for specific dataset
    # NOTE: the names of the variables/observables depend on this 
    # specific dataset && on the 'custom_data.py' script.
    from shared import custom_data
    data_methods = [nm for nm in dir(custom_data) if nm.startswith('process_')]
    nmethods     = len(data_methods)
    var_names    = [ dm.split('process_')[1] for dm in data_methods]
    subparser_.add_argument(
    '-vname', '--vname',
    type=str,
    help='scalar variable to plot. Available options: ' + ', '.join(var_names),
    default=var_names[0],
    )

pa = parser.parse_args()


#--- consistency checks (either individual or massive)
assert not(pa.fname_inp and pa.dir_src), \
    '\n [*] specify either --fname_inp OR --dir_src. Not both.\n'

assert not(pa.fname_fig and pa.dir_dst), \
    '\n [*] specify --fname_fig OR --dir_dst. Not both.'

#--- check we have a valid variable name
assert pa.vname in var_names,\
    ' [-] vname argument should be one of the these: ' + ', '.join(var_names)

#--- check verbose option
if not pa.verbose in ('debug', 'info'):
    raise SystemExit(' [-] Invalid argument: %s\n'%pa.verbose)


# build list of input files
if pa.dir_src:
    assert os.path.isdir(pa.dir_src)
    finp_list = glob.glob(pa.dir_src+'/*.h5')
    assert len(finp_list)>0
    if pa.dir_dst: os.system('mkdir -p ' + pa.dir_dst)
else:
    assert os.path.isfile(pa.fname_inp)
    finp_list = [pa.fname_inp,]


#--- distribute the work
comm          = MPI.COMM_WORLD
rank          = comm.Get_rank() # proc rank
wsize         = comm.Get_size() # nmbr of procs
# list of number of files assigned to each proc
nf_proc       = sf.equi_list(finp_list, wsize)
# nmbr of files assigned to all previous procs (i.e. those ranks < 'rank')
nf_prev       = nf_proc[:rank].sum() 
# list of files assigned to this proc
finp_list_proc = finp_list[nf_prev:nf_prev+nf_proc[rank]]


# grab the name of the module selected in the CLI
mod_selected = sys.argv[1]
print "\n ---> Using %s\n"%mod_selected 
mod = getattr(fp,'cutter__'+mod_selected)()

# build the times series
getattr(mod, 'run')(
    pa, 
    custom_data = custom_data, 
    finp_list_proc = finp_list_proc,
    )

#EOF
