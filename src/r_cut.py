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
'-fi', '--fname_inp',
type=str,
default=None,
help='input ASCII file',
)
parser.add_argument(
'-fig', '--fname_fig',
type=str,
default=None,
help='output PNG file. If --dir_fig or -df is specified, this means the prefix of the output figures.',
)
parser.add_argument(
'-dlon', '--dlon',
type=float,
default=[None,None], #[0., 360.],
nargs=2,
metavar=('LON1','LON2'),
help='interval width in longitude, in degrees.',
)
parser.add_argument(
'-dth', '--dth',
type=float,
default=[None,None], #[-90., 90.],
nargs=2,
metavar=('TH1','TH2'),
help='interval width in theta (co-latitude?), in degrees.',
)
parser.add_argument(
'-ro', '--ro',
type=float,
default=5.0,
help='radius for the spherical shell to be plotted.',
)
parser.add_argument(
'-i', '--interactive',
action='store_true',
default=False,
help='If used, shows an interactive IPython plot; otherwise, it creates a .png figure.',
)
parser.add_argument(
'-v', '--verbose',
type=str,
default='debug',
help='verbosity level (debug=minimal, info=extended)',
)
parser.add_argument(
'-figsize', '--figsize',
type=float,
default=[6,4], #[-90., 90.],
nargs=2,
metavar=('WIDTH','HEIGTH'),
help='figure size',
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

fnm = '/media/scratch1/swmf/IO2__FDIPS_CR2106_iv/IO2/3d__var_1_n00010500.h5'
o = sf.r_cut(fnm, pa.fname_fig, ro=pa.ro, 
    dph=pa.dlon,
    dth=pa.dth,
    figsize=pa.figsize,
    verbose=pa.verbose, 
    vnames=custom_data.vnames, 
    data_processor=getattr(custom_data, 'process_'+pa.vname),
    cscale='log',                   # color scale
    interactive=pa.interactive,
    )




#EOF
