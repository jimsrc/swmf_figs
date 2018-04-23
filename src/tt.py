#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""
from:
/work/ccmc/enlil/06dic14/python/Vr/06dic14_Vr.py
"""
import numpy as np
from numpy import sin, cos, array, sqrt
#---- graphics
from pylab import figure, close, show
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D # necessary for projection='3d' in add_subplot()
import matplotlib.pyplot as plt

#++++++ constants
r2d     = 180./np.pi

#+++++++++++++++++++++++++++++++++++
def calc_phi(_x, _y):
    assert (_x!=0) or (_y!=0.), "\n [-] singularity!\n"

    if ((_x==0.) & (_y>0.)):
        return 0.5*np.pi
    elif ((_x==0.) & (_y<0.)):
        return 1.5*np.pi
    elif (_x<0.):
        return np.pi + np.arctan(_y/_x)
    elif ((_x>0.) & (_y<0.)):
        return 2.*np.pi + np.arctan(_y/_x)
    else:
        return np.arctan(_y/_x)



def read_data(fnm):
    #+++++++++++++++++++++++++++++++++++
    #fnm = '3d__var_1_n00000005.out__chip0_xxxiii.txt'
    x,y,z, bx,by,bz = np.loadtxt(fnm, unpack=1, skiprows=5)

    ndata       = x.size
    r, th, ph   = np.zeros((3,ndata), dtype=np.float)

    r[:]        = np.sqrt(x**2 + y**2 + z**2)
    th[:]       = np.arctan(z / np.sqrt(x**2 + y**2))

    for i in range(ndata):
        #r[i]    = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
        #th[i]   = np.arctan(z[i]/)
        ph[i]   = calc_phi(x[i], y[i])         # [rad]

    # get B modulus
    Bmod = np.sqrt(bx*bx + by*by + bz*bz)

    return r, ph, th, Bmod


def main():
    r, th, ph, ndata = read_data()
    #+++++++++++++++++++++++++++++++++++
    # let's counts the repetitions in spherical coords:
    uniq_r,  counts_r  = np.unique(r,  return_counts=1)
    uniq_th, counts_th = np.unique(th, return_counts=1)
    uniq_ph, counts_ph = np.unique(ph, return_counts=1)

    print np.unique(counts_r)
    print np.unique(counts_th)
    print np.unique(counts_ph)


def deduce_terna(ib, nRootX=8, nRootY=8, nRootZ=4, nc_r=6, nc_ph=4, nc_th=4, nLevel=2):
    """
    According to documentation, X,Y,Z goes in the same order
    as R,LON,LAT.
    """
    # location inside the mem-blk, in units of sub-blocks
    # TODO: chekar si hay q reemplazar con 'nLevel'
    # TODO: probar con otros valores de nLevel!=2
    ibb = ib % (nLevel**3)
    ibb_r   = int((ibb % (2**1))/(2**0))
    ibb_ph  = int((ibb % (2**2))/(2**1))
    ibb_th  = int((ibb % (2**3))/(2**2))

    # the "octree" is composed by nLevel**3 children-blocks.
    ib_mem      = int(ib/(nLevel**3))
    ib_mem_r    = ib_mem % nRootX
    ib_mem_ph   = (ib_mem/nRootX) % nRootY
    ib_mem_th   = ib_mem / (nRootX*nRootY)

    return {
        'ib_mem':   [ib_mem_r, ib_mem_ph, ib_mem_th], 
        'ibb':      [ibb_r, ibb_ph, ibb_th],
        }


def coord_of_terna(ib_mem=[0,0,0], ibb=[0,0,0], nRootX=8, nRootY=8, nRootZ=4, nc_r=6, nc_ph=4, nc_th=4, nLevel=2):
    """
    return 1D-coordinate of the initial position of 
    the memory-block = [ib_mem_r, ib_mem_ph, ib_mem_th]
    """
    ib_mem_r, ib_mem_ph, ib_mem_th = ib_mem
    assert ((ib_mem_r<nRootX) & (ib_mem_ph<nRootY) & (ib_mem_th<nRootZ)),\
        ' [-] ERROR.\n'

    # the initial cardinal position of the memory-block in
    # units of sub-blocks
    ib_coord = ib_mem_r  * (nLevel**3) +\
            ib_mem_th * (nLevel**3) * nRootX +\
            ib_mem_ph * (nLevel**3) * (nRootX*nRootZ)

    # locate the point inside the memory-block
    ibb_r, ibb_ph, ibb_th   = ibb

    # TODO: chekar si hay q reemplazar '2' con 'nLevel'
    # sub-block id 
    assert all(np.array([ibb_r,ibb_ph,ibb_th]) < nLevel), ' [-] ERROR!\n'
    card_inside_blk = ibb_r*(2**0) + ibb_ph*(2**2) + ibb_th*(2**1)
    card_subblk = card_inside_blk + ib_coord

    # cardinal inside the memory-block
    card_cell   = card_subblk * (nc_r*nc_ph*nc_th)
    card        = card_cell + ib_coord*(nc_r*nc_ph*nc_th)

    return {
        'ib_coord'      : ib_coord,
        'card_subblk'   : card_subblk,
        'card_cell'     : card_cell,
        }



def iterate():
    r, th, ph, ndata        = read_data()
    nLevel                  = 2
    nc_r, nc_ph, nc_th      = 6, 4, 4
    nRootX, nRootY, nRootZ  = 8, 8, 4
    nb_r, nb_ph, nb_th      = nLevel*nRootX, nLevel*nRootY, nLevel*nRootZ
    # check...
    assert nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th == ndata
    
    np.set_printoptions(precision=2, linewidth=200)
    #for i in range(9*4, 19*4):
    #for i in range(8*7*4, 8*7*4 + 9*4):
    _nlayer = nc_r*nc_th
    for _ib in range(0,10):
        icell   = _ib*(nc_r*nc_th*nc_ph)
        ib_loc  = deduce_terna(_ib)
        print " -->  ", ib_loc['ib_mem'], ib_loc['ibb']

        print "%d; ib=%d" % (icell/_nlayer, _ib)
        # show a "layer" of cells in (r,th)
        print  r[icell:icell+_nlayer]
        print ph[icell:icell+_nlayer]
        print th[icell:icell+_nlayer]


    print "++++++++++++++++++++++++++++"
    n = nc_r*nc_ph*nc_th * (nb_r-1)
    print r[n-nc_r:n+nc_r]
    print ph[n-nc_r:n+nc_r]
    print th[n-nc_r:n+nc_r]

    # this shows the periodicity...
    print "\n [+] Just a check: \n"
    print r[0], r[nc_r*nc_ph*nb_ph*nb_r]
    print r[nc_r*nc_ph*nb_ph*nb_r-2], r[nc_r*nc_ph*nb_ph*nb_r-1], r[nc_r*nc_ph*nb_ph*nb_r+1], r[nc_r*nc_ph*nb_ph*nb_r+2]


def rearrange():
    fnm = '3d__var_1_n00000005.out__chip0_xxxiii.txt'
    r, ph, th, ndata        = read_data(fnm)
    nLevel                  = 2
    nc_r, nc_ph, nc_th      = 6, 4, 4
    nRootX, nRootY, nRootZ  = 8, 8, 4
    # nmbr of blocks (or "sub-blocks")
    nb_r, nb_ph, nb_th      = nLevel*nRootX, nLevel*nRootY, nLevel*nRootZ
    assert nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th == ndata

    # we'll build this
    coords  = np.zeros((nc_r*nb_r, nc_ph*nb_ph, nc_th*nb_th, 3))

    ib = 0 
    while ib*(nc_r*nc_th*nc_ph) < r.size:
        for ic_th in range(nc_th):
            for ic_ph in range(nc_ph):
                for ic_r in range(nc_r):
                    # location of the sub-block
                    ib_mem_r,ib_mem_ph,ib_mem_th = deduce_terna(ib,)['ib_mem']
                    # local  location in units of sub-blocks
                    ibb_r, ibb_ph, ibb_th        = deduce_terna(ib,)['ibb']
                    # global location in units of sub-blocks
                    ib_r    = ib_mem_r *nLevel + ibb_r
                    ib_ph   = ib_mem_ph*nLevel + ibb_ph
                    ib_th   = ib_mem_th*nLevel + ibb_th
                    # global cardinal index
                    ind =   ic_r + ic_ph*nc_r + ic_th*(nc_r*nc_ph) +\
                            ib*(nc_r*nc_th*nc_ph)
                    ind_r   = ic_r  + ib_r *nc_r
                    ind_ph  = ic_ph + ib_ph*nc_ph
                    ind_th  = ic_th + ib_th*nc_th
                    # info on screen
                    print " block:%d/%d, %d/%d ,%d/%d; " %\
                        (ib_r,  nRootX*nLevel, \
                         ib_ph, nRootY*nLevel, \
                         ib_th, nRootZ*nLevel),
                    print ';    cell:%d,%d,%d;  ind:%06d' % \
                        (ic_r,ic_ph,ic_th, ind),
                    print ';   (r,ph,th)=(%.2f, %.2f, %.2f)' % \
                        (r[ind], ph[ind]*r2d, th[ind]*r2d)
                    # check mem block indexes
                    assert ib_mem_r  < nRootX
                    assert ib_mem_ph < nRootY
                    assert ib_mem_th < nRootZ
                    # check sub-block indexes
                    assert ibb_r  < nLevel
                    assert ibb_ph < nLevel
                    assert ibb_th < nLevel
                    # check final indexes
                    assert ind_r  < nc_r  * (nRootX*nLevel)
                    assert ind_ph < nc_ph * (nRootY*nLevel)
                    assert ind_th < nc_th * (nRootZ*nLevel)
                    # if it's not in zeros, you're trying to overwrite something!
                    assert all(coords[ind_r,ind_ph,ind_th,:] == [0,0,0]),\
                        '\n [-] ERROR: trying to overwrite values!\n'
                    coords[ind_r,ind_ph,ind_th,:] = r[ind], ph[ind], th[ind]
        ib += 1

    return coords


def get_array_Bmod(fname_inp, nc=[6,4,4], nRoot=[8,8,4], nLevel=2):
    r, ph, th, Bmod         = read_data(fname_inp)
    ndata                   = r.size
    nc_r, nc_ph, nc_th      = nc
    nRootX, nRootY, nRootZ  = nRoot
    # nmbr of blocks (or "sub-blocks")
    nb_r, nb_ph, nb_th      = nLevel*nRootX, nLevel*nRootY, nLevel*nRootZ
    assert nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th == ndata

    # initialize data
    data            = np.zeros((nc_r*nb_r, nc_ph*nb_ph, nc_th*nb_th))
    _r              = np.zeros(nc_r*nb_r)
    _ph             = np.zeros(nc_ph*nb_ph)
    _th             = np.zeros(nc_th*nb_th)

    ib = 0 
    while ib*(nc_r*nc_th*nc_ph) < r.size:
        print " block: ", ib
        for ic_th in range(nc_th):
            for ic_ph in range(nc_ph):
                for ic_r in range(nc_r):
                    # location of the sub-block
                    ib_mem_r,ib_mem_ph,ib_mem_th = deduce_terna(ib,)['ib_mem']
                    # local  location in units of sub-blocks
                    ibb_r, ibb_ph, ibb_th        = deduce_terna(ib,)['ibb']
                    # global location in units of sub-blocks
                    ib_r    = ib_mem_r *nLevel + ibb_r
                    ib_ph   = ib_mem_ph*nLevel + ibb_ph
                    ib_th   = ib_mem_th*nLevel + ibb_th
                    # global cardinal index
                    ind =   ic_r + ic_ph*nc_r + ic_th*(nc_r*nc_ph) +\
                            ib*(nc_r*nc_th*nc_ph)
                    ind_r   = ic_r  + ib_r *nc_r
                    ind_ph  = ic_ph + ib_ph*nc_ph
                    ind_th  = ic_th + ib_th*nc_th
                    # info on screen
                    #print " block:%d/%d, %d/%d ,%d/%d; " %\
                    #    (ib_r,  nRootX*nLevel, \
                    #     ib_ph, nRootY*nLevel, \
                    #     ib_th, nRootZ*nLevel),
                    #print ';    cell:%d,%d,%d;  ind:%06d' % \
                    #    (ic_r,ic_ph,ic_th, ind),
                    #print ';   (r,ph,th)=(%.2f, %.2f, %.2f)' % \
                    #    (r[ind], ph[ind]*r2d, th[ind]*r2d)
                    # check mem block indexes
                    assert ib_mem_r  < nRootX
                    assert ib_mem_ph < nRootY
                    assert ib_mem_th < nRootZ
                    # check sub-block indexes
                    assert ibb_r  < nLevel
                    assert ibb_ph < nLevel
                    assert ibb_th < nLevel
                    # check final indexes
                    assert ind_r  < nc_r  * (nRootX*nLevel)
                    assert ind_ph < nc_ph * (nRootY*nLevel)
                    assert ind_th < nc_th * (nRootZ*nLevel)
                    # if it's not in zeros, you're trying to overwrite something!
                    assert (data[ind_r,ind_ph,ind_th] == 0.0),\
                        '\n [-] ERROR: trying to overwrite values!\n'
                    data[ind_r,ind_ph,ind_th] = Bmod[ind]
                    _r[ind_r]   = r[ind]
                    _ph[ind_ph] = ph[ind]
                    _th[ind_th] = th[ind]
        ib += 1

    return {
    'coords'    : [_r, _ph, _th],
    'Bmod'      : data,
    }


def contour_2d(fig, ax, x, y, mat, hscale='log', **kargs):
    cb_label = kargs.get('cb_label', 'points per bin square')
    cb_fontsize = kargs.get('cb_fontsize', 15)
    cbmin, cbmax = kargs.get('vmin',1), kargs.get('vmax',1e3)
    opt = {
    'linewidth': 0,
    'cmap': cm.gray_r,                # gray-scale
    'vmin': cbmin, #kargs.get('cbmin',1),
    'vmax': cbmax, #kargs.get('cbmax',1000),
    'alpha': kargs.get('alpha',0.9),
    }
    if hscale=='log':
        opt.update({'norm': LogNorm(),})
    #--- 2d contour
    surf = ax.contourf(x, y, mat, facecolors=cm.jet(mat), **opt)
    sm = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
    sm.set_array(mat)
    #--- colorbar
    axcb = fig.colorbar(sm)
    axcb.set_label(cb_label, fontsize=cb_fontsize)
    sm.set_clim(vmin=cbmin, vmax=cbmax)
    return fig, ax


def get_index_r(r, ro):
    dr_max  = (r[1:] - r[:-1]).max() # coarser resolution in r

    # what follows is valid if we fulfill this:
    rmin, rmax = r[0], r[-1]
    assert (ro<=rmax) and (ro>=rmin),\
        '\n [-] ERROR: \'ro\' must be inside the interval (%g, %g)\n'%(rmin,rmax)

    ro_behind, ro_ahead = r[r<=ro][-1], r[r>=ro][0]
    be_gt_ah    = (ro-ro_behind) >= (ro_ahead-ro)
    if be_gt_ah:
        i_r = (r>=ro).nonzero()[0][0] ## RIGHT??. Check first!!
    else:
        i_r = (r<=ro).nonzero()[0][-1]

    return i_r


if __name__=='__main__':
    import argparse
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

    import numpy as np
    np.set_printoptions(precision=2, linewidth=230)
    r2d         = 180./np.pi

    fnm         = '3d__var_1_n00000005.out__chip0_xxxiii.txt'
    d           = get_array_Bmod(fnm, nc=[6,4,4], nRoot=[8,8,4], nLevel=2)
    r, ph, th   = d['coords']
    Bmod        = d['Bmod']; 
    # filter 
    cc          = ph == ph[10]

    #--- figure
    fig     = figure(1,)
    ax      = fig.add_subplot(111, projection='3d')
    # mesh versions of the coords
    R, PH, TH = np.meshgrid(r, ph, th)
    # get the cartesian coords (I know!)
    X = R * np.cos(TH) * np.cos(PH)
    Y = R * np.cos(TH) * np.sin(PH)
    Z = R * np.sin(TH)

    #--- slice an specific shell r=ro
    i_r = 1
    i_r = get_index_r(r, pa.ro)
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
