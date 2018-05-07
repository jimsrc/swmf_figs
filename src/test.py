#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
import shared.funcs as sf
from mpi4py import MPI

#--- retrieve args
parser = argparse.ArgumentParser(
formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
'-clim', '--clim',
type=float,
nargs=2,
default=[None, None],
help='colorbar limits',
)
parser.add_argument(
'-cs', '--cb_scale',
type=str,
default='log', # 'log'
help='colorbar scale ("linear" or "log")',
)
parser.add_argument(
'-fi', '--fname_inp',
type=str,
default=None,
help='input ASCII file. If this option is used, don\'t use --dir_src.',
)
parser.add_argument(
'-ff', '--fname_fig',
type=str,
default=None,
help='output PNG file. If --dir_dst is specified, this option must not be used.',
)
parser.add_argument(
'-ds', '--dir_src',
type=str,
default=None,
help='input directory. If this option is used, don\'t use --fname_inp.',
)
parser.add_argument(
'-dd', '--dir_dst',
type=str,
default=None,
help='input directory',
)
parser.add_argument(
'-ro', '--ro',
type=float,
default=5.0,
help='radius for the spherical shell to be plotted.',
)
parser.add_argument(
'-pho', '--pho',
type=float,
default=0.0,
help='longitude value for the cut',
)
parser.add_argument(
'-dlon', '--dlon',
type=float,
default=0.0,
help='interval width for the cut in longitude',
)
parser.add_argument(
'-rr', '--r_range',
type=float,
nargs=2,
default=[2., 5.],
help='radius for the spherical shell to be plotted.',
)
parser.add_argument(
'-v', '--verbose',
type=str,
default='debug',
help='verbosity level (debug=minimal, info=extended)',
)
parser.add_argument(
'-pazim', '--pazim', #action='store',
type=float,
default=-60.,
help='perspective azimuth',
)
parser.add_argument(
'-dpi', '--dpi', #action='store',
type=int,
default=135,
help='dots per inch for savefig()',
)
# import custom methods for specific dataset
# NOTE: the names of the variables/observables depend on this 
# specific dataset && on the 'custom_data.py' script.
from shared import custom_data
data_methods = [nm for nm in dir(custom_data) if nm.startswith('process_')]
nmethods     = len(data_methods)
var_names    = [ dm.split('process_')[1] for dm in data_methods]
parser.add_argument(
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
    print " >>> creating "+pa.dir_dst
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


#--- build figs
for finp in finp_list_proc:
    if not pa.fname_inp:    # massive mode
        fname_fig = finp.replace('.h5','__'+pa.vname+'.png')
        if pa.dir_dst:
            # change the dir path
            fname_fig = pa.dir_dst + '/' + fname_fig.split('/')[-1]
    else:                   # individual mode
        fname_fig = pa.fname_fig

    sf.make_3dplot(
        finp, 
        fname_fig, 
        pa.clim,
        vnames = custom_data.vnames, 
        data_processor = getattr(custom_data, 'process_'+pa.vname),
        verbose = pa.verbose, 
        ro = pa.ro,
        pho = pa.pho,
        r_range = pa.r_range,
        pazim = pa.pazim,
        cscale = pa.cb_scale,         # colorbar scale
        dpi = pa.dpi,
        wtimelabel = True,
    )
"""
#--- w hdf5 input
sf.make_3dplot_hdf5(pa.fname_inp, pa.fname_fig, pa.clim,
    vname = pa.vname,
    verbose=pa.verbose, 
    ro=pa.ro,
    pazim = pa.pazim,
    cscale=pa.cb_scale,         # colorbar scale
)
"""

#EOF
