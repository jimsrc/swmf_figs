#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
import argparse
import numpy as np
import funcs as ff
from pylab import figure, close
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D # necessary for projection='3d' in add_subplot()


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
'-ro', '--ro', #action='store',
type=float,
default=5.,
help='radius for the slice to plot',
)
pa = parser.parse_args()


np.set_printoptions(precision=2, linewidth=230)
r2d         = 180./np.pi

fnm         = '3d__var_1_n00000005.out__chip0_xxxiii.txt'
d           = ff.get_array_Bmod(fnm, nc=[6,4,4], nRoot=[8,8,4], nLevel=2)
r, ph, th   = d['coords']
Bmod        = d['Bmod']; 
# filter 
cc          = ph == ph[10]

#--- figure
fig     = figure(1,)
ax      = fig.add_subplot(111, projection='3d')

# mesh versions of the coords
R, PH, TH   = np.meshgrid(r, ph, th)
# get the cartesian coords (I know!)
X           = R * np.cos(TH) * np.cos(PH)
Y           = R * np.cos(TH) * np.sin(PH)
Z           = R * np.sin(TH)

#--- slice an specific shell r=ro
i_r = ff.get_index_r(r, pa.ro)
print ' > We\'ll plot i_r: ', i_r

# we need the transpose in order to be consistent with 'plot_surface'
var = Bmod.transpose((1,0,2))[:,i_r,:]
#print ' >> ', Bmod.min(), Bmod.max()
# NOTE: 'plot_surface' can only plot variables with shape (n,m), so 
# no 3D variables.
cbmin, cbmax = [var.min(), var.max()] if pa.clim==[None,None] else pa.clim
print " >> ", var.min(), var.max(), var.mean(), np.median(var)

norm = Normalize(cbmin, cbmax)
# other options
opt = {
'rstride'       : 1,
'cstride'       : 1,
'linewidth'     : 0,
'antialiased'   : False,
'shade'         : False,
'alpha'         : 1., #kargs.get('alpha',0.9),
'cmap'          : cm.jet,                # gray-scale
'norm'          : Normalize(cbmin, cbmax),
'vmin'          : cbmin, #kargs.get('cbmin',1),
'vmax'          : cbmax, #kargs.get('cbmax',1000),
}
surf = ax.plot_surface(X[:,i_r,:], Y[:,i_r,:], Z[:,i_r,:], 
    facecolors=cm.jet(norm(var)), **opt)
# Note the cm.jet(..) --> cm.jet(norm(..)); see:
# https://stackoverflow.com/questions/25023075/normalizing-colormap-used-by-facecolors-in-matplotlib
# perspective azimuth
ax.azim = pa.pazim
sm = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
sm.set_array(var); #surf.set_array(var)

ax.set_xlabel('X [Ro]')
ax.set_ylabel('Y [Ro]')
ax.set_zlabel('Z [Ro]')
ax.set_title('$r_o$ = %.2g $R_o$' % r[i_r])

#--- colorbar
cb_label = '|B| [G]'
cb_fontsize = 13
axcb = fig.colorbar(sm, ax=ax)
axcb.set_label(cb_label, fontsize=cb_fontsize)
sm.set_clim(vmin=cbmin, vmax=cbmax)

# save figure
#show()
fig.savefig('test.png', dpi=135, bbox_inches='tight')
close(fig)


#EOF
