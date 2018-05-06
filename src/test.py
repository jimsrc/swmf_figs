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
'-cs', '--cb_scale',
type=str,
default='log', # 'log'
help='colorbar scale ("linear" or "log")',
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

#--- check we have a valid variable name
assert pa.vname in var_names,\
    ' [-] vname argument should be one of the these: ' + ', '.join(var_names)

#--- check verbose option
if not pa.verbose in ('debug', 'info'):
    raise SystemExit(' [-] Invalid argument: %s\n'%pa.verbose)

sf.make_3dplot(pa.fname_inp, pa.fname_fig, pa.clim,
    vnames=custom_data.vnames, 
    data_processor=getattr(custom_data, 'process_'+pa.vname),
    verbose=pa.verbose, 
    ro=pa.ro,
    pho = pa.pho,
    r_range=pa.r_range,
    pazim = pa.pazim,
    cscale=pa.cb_scale,         # colorbar scale
    dpi=pa.dpi,
    wtimelabel=True,
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
