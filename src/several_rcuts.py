#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import argparse, os, glob
import numpy as np
import shared.funcs as sf

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
'-pazim', '--pazim', #action='store',
type=float,
default=-60.,
help='perspective azimuth',
)
parser.add_argument(
'-nLevel', '--nLevel', #action='store',
type=int,
default=2,
help='nLevel from #GRIDLEVEL of PARAM.in',
)
parser.add_argument(
'-fi', '--fname_inp',
type=str,
default=None,
help='input ASCII file',
)
parser.add_argument(
'-ff', '--fname_fig',
type=str,
default=None,
help='output PNG file. If --dir_fig or -df is specified, this means the prefix of the output figures.',
)
parser.add_argument(
'-ds', '--dir_src',
type=str,
default=None,
help='directory where input ASCII files (*.out) are placed',
)
parser.add_argument(
'-df', '--dir_fig',
type=str,
default=None,
help='directory where output figures will be paced',
)
parser.add_argument(
'-c', '--checks',
action='store_true',
default=False,
help='checks size consitencies in the number of entries of the input file',
)
parser.add_argument(
'-ro', '--ro',
type=float,
default=5.0,
help='radius for the spherical shell to be plotted.',
)
parser.add_argument(
'-v', '--verbose',
type=str,
default='debug',
help='verbosity level (debug=minimal, info=extended)',
)
# import custom methods for specific dataset
# NOTE: the names of the variables/observables depend on this 
# specific dataset && on the 'custom_data.py' script.
import custom_data
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

#--- check we have a valid variable name
assert pa.vname in var_names,\
    ' [-] vname argument should be one of the these: ' + ', '.join(var_names)

#--- check verbose option
if not pa.verbose in ('debug', 'info'):
    raise SystemExit(' [-] Invalid argument: %s\n'%pa.verbose)

#--- check consistency in the input data
ok_single  = (pa.fname_inp is not None) and \
    (pa.dir_src is None  and  pa.dir_fig is None)
ok_massive = (pa.fname_inp is None) and \
    (pa.dir_src is not None  and  pa.dir_fig is not None)
assert ok_single or ok_massive, \
    ' [-] ERROR: you should either specify the input filename or the input diretory!\n'


if pa.fname_inp is not None:
    # plot ONE snapshot
    o = sf.plot_sphere_cuts(pa.fname_inp, pa.fname_fig, ro=pa.ro, 
        pazim=pa.pazim, clim=pa.clim, checks=pa.checks, complete=True,
        verbose=pa.verbose, vnames=custom_data.vnames, 
        data_processor=getattr(custom_data, 'process_'+pa.vname),
        nc=None, nRoot=None, nLevel=None,
    )
elif pa.dir_src is not None:
    # plot SEVERAL snapshots
    assert os.path.isdir(pa.dir_src), \
        '[-] the directory \'%s\' does not exist!\n' % pa.dir_src
    sf.make_sphere_shells(pa.dir_src, pa.dir_fig, prefix_fig=pa.fname_fig, 
        ro=pa.ro, pazim=pa.pazim, clim=pa.clim, verbose=pa.verbose, 
        vnames=custom_data.vnames,
        data_processor=getattr(custom_data, 'process_'+pa.vname),
        )
else:
    raise SystemExit(' [-] wrong inputs!\n')

#EOF
