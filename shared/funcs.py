#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""
from:
/work/ccmc/enlil/06dic14/python/Vr/06dic14_Vr.py
"""
import numpy as np
#from numpy import sin, cos, array, sqrt
import logging, argparse, glob
import h5py
#---- system
from subprocess import Popen, PIPE, STDOUT
import re
#---- graphics
from pylab import figure, close, show
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D # necessary for projection='3d' in add_subplot()
#---- polar plots (w/ partial limits)
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.angle_helper as angle_helper
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import (FixedLocator, MaxNLocator,
                                                 DictFormatter)


#++++++ constants
r2d     = 180./np.pi
np.set_printoptions(precision=2, linewidth=230)

#++++++ logging
logger  = logging.getLogger()
logger.setLevel(logging.DEBUG) # 'DEBUG' by default
ch = logging.StreamHandler()
# logging setup
formatter = logging.Formatter("# %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)



def equi_list(flist, npart):
    """
    returns most equi-partitioned tuple of lists 
    of days between date-objects 'dini' and 'dend'
    """
    nf      = len(flist)
    nf_part = np.zeros(npart, dtype=np.int)
    resid   = np.mod(nf, npart)
    for i in range(npart-resid):
        nf_part[i] = nf/npart
    # last positions where I put residuals
    last = np.arange(start=-1,stop=-resid-1,step=-1)
    for i in last:
        nf_part[i] = nf/npart + 1

    assert np.sum(nf_part)==nf, \
        " --> somethng went wrong!  :/ "

    return nf_part

def equi_days(dini, dend, n):
    """
    returns most equi-partitioned tuple of number 
    of days between date-objects 'dini' and 'dend'
    """
    days = (dend - dini).days
    days_part = np.zeros(n, dtype=np.int)
    resid = np.mod(days, n)
    for i in range(n-resid):
        days_part[i] = days/n
    # last positions where I put residuals
    last = np.arange(start=-1,stop=-resid-1,step=-1)
    for i in last:
        days_part[i] = days/n+1

    assert np.sum(days_part)==days, \
        " --> somethng went wrong!  :/ "

    return days_part

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


#@profile
def read_data(fnm, vnames):
    """
    fnm     : input filename
    vnames  : variable names
    odata   : output data (processed from ASCII data)
    """
    print "\n [+] reading data...\n"
    fformat = file_format(fnm)
    vdict = {}
    if fformat=='ascii':
        inp = np.loadtxt(fnm, unpack=True, skiprows=5)
        # make sure we match the number of fields in the ASCII file
        assert len(inp)==(3 + len(vnames)), \
            '\n [-] vnames doesn\'t match the number of fields in the ASCII file!\n'
        # get the coordinates
        x, y, z = inp[:3]
        # get output variables from simulation
        for nm, _inm in zip(vnames, range(len(vnames))):
            vdict[nm] = inp[3+_inm]
        print ' [*] read ok.\n'

        ndata       = x.size
        r, th, ph   = np.zeros((3,ndata), dtype=np.float)

        r[:]        = np.sqrt(x**2 + y**2 + z**2)
        th[:]       = np.arctan(z / np.sqrt(x**2 + y**2))

        for i in range(ndata):
            #r[i]    = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
            #th[i]   = np.arctan(z[i]/)
            ph[i]   = calc_phi(x[i], y[i])         # [rad]

    elif fformat=='hdf5':
        logger.info(' [*] reading an HDF5 file...')
        f5  = h5py.File(fnm, 'r')
        r   = f5['coords/r'][...]
        ph  = f5['coords/ph'][...]
        th  = f5['coords/th'][...]
        for nm in f5['data'].keys():
            # don't read variables that we did not ask for.
            if nm not in vnames: pass
            assert nm not in vdict.keys(),\
                '\n [-] vdict already has the key! (%s)\n' % nm
            logger.debug(' [*] reading variable: '+nm)
            vdict[nm] = f5['data/'+nm][...]
        assert len(vdict.keys()) > 0,\
        ' [-] we did not grab any variable from:\n %r\n by using the parsed list:\n %r\n'% (f5.keys(), vnames)
        f5.close()

    elif fformat=='binary': # IDL binary
        import spacepy.pybats as pb
        # NOTE: SpacePy knows how to handle IDL binary!
        fpb = pb.IdlFile(fnm, format=fformat)

        # get coordinates
        x, y, z = fpb['x'][...], fpb['y'][...], fpb['z'][...]

        # get output variables from simulation
        #fkeys_low = [ _k.lower() for _k in fpb.keys() ]
        assert all([_vnm in fpb.keys() for _vnm in vnames])
        for nm in vnames:
            vdict[nm] = fpb[nm][...]
        print ' [*] read ok.\n'

        ndata       = x.size
        r, th, ph   = np.zeros((3,ndata), dtype=np.float)

        r[:]        = np.sqrt(x**2 + y**2 + z**2)
        th[:]       = np.arctan(z / np.sqrt(x**2 + y**2))
        ph[:]       = [ calc_phi(x[i], y[i]) for i in range(ndata) ]

        assert len(vdict.keys()) > 0,\
        ' [-] we did not grab any variable from:\n %r\n by using the parsed list:\n %r\n'% (fpb.keys(), vnames)
        # TODO: need to close 'fpb'?
        del fpb

    else:
        raise SystemExit('\n [-] wrong file format: %r\n'%fformat)

    return r, ph, th, vdict, fformat


def ib_to_ind(ic, ib, coords, nRoot, nc, nLevel):
    ic_r, ic_ph, ic_th      = ic
    r, ph, th               = coords
    nc_r, nc_ph, nc_th      = nc
    nRootX, nRootY, nRootZ  = nRoot
    npart_D = 2**nLevel # number of bi-partitions in EACH DIMENSION
    nb_r                    = npart_D*nRootX
    nb_ph                   = npart_D*nRootY
    nb_th                   = npart_D*nRootZ

    _ibloc = deduce_terna(ib, nRoot, nc, nLevel)
    # location of the sub-block
    ib_mem_r,ib_mem_ph,ib_mem_th = _ibloc['ib_mem']
    # local  location in units of sub-blocks
    ibb_r, ibb_ph, ibb_th        = _ibloc['ibb']
    # global location in units of sub-blocks
    ib_r    = ib_mem_r *npart_D + ibb_r
    ib_ph   = ib_mem_ph*npart_D + ibb_ph
    ib_th   = ib_mem_th*npart_D + ibb_th
    # global cardinal index
    ind     = ic_r + ic_ph*nc_r + ic_th*(nc_r*nc_ph) +\
                ib*(nc_r*nc_th*nc_ph)
    ind_r   = ic_r  + ib_r *nc_r
    ind_ph  = ic_ph + ib_ph*nc_ph
    ind_th  = ic_th + ib_th*nc_th
    # info on screen
    print " block:%d/%d, %d/%d, %d/%d " %\
        (ib_r,  nb_r, \
         ib_ph, nb_ph, \
         ib_th, nb_th),
    #if ib_th >= 6:
    #    import pdb; pdb.set_trace()
    print ';    cell:%d,%d,%d;  ind:%06d (%d,%d,%d)' % \
        (ic_r, ic_ph, ic_th, ind, ind_r, ind_ph, ind_th),
    print ';   (r,ph,th)=(%.2f, %.2f, %.2f)' % \
        (r[ind], ph[ind]*r2d, th[ind]*r2d)
    # check mem block indexes
    assert ib_mem_r  < nRootX
    assert ib_mem_ph < nRootY
    assert ib_mem_th < nRootZ
    # check sub-block indexes
    assert ibb_r  < npart_D
    assert ibb_ph < npart_D
    assert ibb_th < npart_D
    # check final indexes
    assert ind_r  < nc_r  * (nRootX*npart_D)
    assert ind_ph < nc_ph * (nRootY*npart_D)
    assert ind_th < nc_th * (nRootZ*npart_D)

    return ind, ind_r, ind_ph, ind_th

#@profile
def get_array_vars(fname_inp=None, data=None, checks=False, complete_domain_walk=False, vnames=[], data_processor=None, vdict=None, vectorial=False):
    """
    - read data from the ASCII file w/o assuming that it is consistent
      w/ a complete grid-structure of cells and children-blocks
    - We just produce one SCALAR observable determined by the
      function 'data_processor()'.
    """
    assert (fname_inp is not None) or (data is not None),\
        ' [-] ERROR: we need ASCII/HDF5 file or parsed data!\n'

    if data is None:
        # read_data() will handle the file format
    	r, ph, th, _vdict, fformat = read_data(fname_inp, vnames)
    else:
        r, ph, th, _vdict, fformat = data

    if vdict is not None: vdict.update(_vdict)
    ndata            = r.size

    # obtain Bmod from vdict
    logger.info(' [+] processing ASCII data to obtain observable...')
    assert data_processor is not None, ' [-] We need a processor function!\n'
    
    if fformat in ('ascii', 'binary'): # we need some treatment
        # NOTE: in case vectorial==True, we need the transpose.
        pdata = np.transpose(data_processor(_vdict))
        if not(len(pdata.shape)==1 and not vectorial) and \
            not(len(pdata.shape)>1 and vectorial):
            raise SystemExit(' [-] parser conflict: observables is either scalar or vectorial.\n')
        # make a full-domain discovery by walking all the
        # entries (one by one) in the ASCII file.
        eps = 0.005 # tolerance for degeneration detection
        logger.info(' [+] making domain discovery...')
        _r, _ph, _th = get_domains([r,ph,th],
            eps=eps,  # precision for domain discovery
            checks=checks,
            complete=complete_domain_walk, 
            nc=None, nRoot=None, nLevel=None,
            )
        print('')
        logger.info(' [+] domain is in the ranges: ')
        logger.info('     r  in (%g, %g) '   % ( _r.min(), _r.max()))
        logger.info('     ph in (%g, %g) '   % (_ph.min(), _ph.max()))
        logger.info('     th in (%g, %g) \n' % (_th.min(), _th.max()))

        # allocate data buffer 
        if len(pdata.shape) == 1:   # scalar
            data = np.nan * np.ones((_r.size,_ph.size,_th.size),    dtype=np.float32)
        else:                       # vector
            data = np.nan * np.ones((_r.size,_ph.size,_th.size, 3), dtype=np.float32)
        
        logger.info(' [+] building 3D array...')
        for ind in range(ndata):
            # find the coordinate where it coincides with any 
            # of the (_r, _ph, _th)
            i_r     = (np.abs(_r  -  r[ind]) < eps).nonzero()[0][0]
            i_ph    = (np.abs(_ph - ph[ind]) < eps).nonzero()[0][0]
            i_th    = (np.abs(_th - th[ind]) < eps).nonzero()[0][0]
           
            # make sure we are assignating values to this array element
            # for the 1st time!
            assert np.all(np.isnan(data[i_r,i_ph,i_th])), \
                '\n [-] this array element already has a value!!\n'
            # assignate value to this array-element
            data[i_r,i_ph,i_th] = pdata[ind]
  
        fill_perc = 100.*(data.size - np.isnan(data).nonzero()[0].size)/data.size
        logger.info(' [+] the data array was filled at %g %% \n' % fill_perc)
    elif fformat=='hdf5':
        _r, _ph, _th = r, ph, th
        data         = data_processor(_vdict)
        # NOTE: in case vetorial==True, we need a transpose
        if vectorial:
            data = data.transpose((1,2,3,0))
    else:
        raise SystemExit(' [-] wrong format (%r)!'%fformat)

    return {
    'ndata'     : ndata,
    'coords'    : (_r, _ph, _th),
    'data'      : data
    }

#@profile
def get_array_Bmod(fname_inp, nc=[6,4,4], nRoot=[8,8,4], nLevel=1):
    r, ph, th, Bmod         = read_data(fname_inp)
    ndata                   = r.size
    nc_r, nc_ph, nc_th      = nc
    nRootX, nRootY, nRootZ  = nRoot
    print "------ comeon men..."
    npart_D = 2**nLevel # number of bi-partitions in EACH DIMENSION
    # nmbr of blocks (or "sub-blocks")
    nb_r                    = npart_D*nRootX
    nb_ph                   = npart_D*nRootY
    nb_th                   = npart_D*nRootZ
    if nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th == ndata:
        print ' [+] the number of entries in the file is consistent'
        print '     with the number of cells/blocks and the nLevel'
        print '     parameters!'
    else:
        # inconsistency in number of entries!
        raise SystemExit("""
        Inconsistency in the number of entries in the file!
        >> expected                  : %d
        >> number of entries in file : %d
        """%(nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th, ndata))

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
                    ind, ind_r, ind_ph, ind_th = \
                        ib_to_ind([ic_r,ic_ph,ic_th], ib, [r,ph,th], 
                            nRoot, nc, nLevel)

                    # if it's not in zeros, you're trying to overwrite something!
                    assert (data[ind_r,ind_ph,ind_th] == 0.0),\
                        '\n [-] ERROR: trying to overwrite values!\n'
                    if _r[ind_r]!=0.0 and np.abs(r[ind]-_r[ind_r])>0.005:
                        import pdb; pdb.set_trace()
                    data[ind_r,ind_ph,ind_th] = Bmod[ind]
                    _r[ind_r]   = r[ind]
                    _ph[ind_ph] = ph[ind]
                    _th[ind_th] = th[ind]
        ib += 1

    return {
    'coords'    : [_r, _ph, _th],
    'Bmod'      : data,
    }

#@profile
def get_index_r(r, ro):
    """
    NOTE: we assume 'r' is a monotonically ascending variable.
    """
    dr_max  = (r[1:] - r[:-1]).max() # coarser resolution in r

    # what follows is valid if we fulfill this:
    rmin, rmax = r[0], r[-1]
    assert (ro<=rmax) and (ro>=rmin),\
        '\n [-] ERROR: \'ro\' must be inside the interval (%g, %g)\n'%(rmin,rmax)
    assert r[0]==r.min() and r[-1]==r.max(), \
        ' [-] ERROR: this variable should be monotically ascending!\n'

    ro_behind, ro_ahead = r[r<=ro][-1], r[r>=ro][0]
    be_gt_ah    = (ro-ro_behind) >= (ro_ahead-ro)
    if be_gt_ah:
        i_r = (r>=ro).nonzero()[0][0] ## RIGHT??. Check first!!
    else:
        i_r = (r<=ro).nonzero()[0][-1]

    return i_r

def get_subcoord(isb, iLevel=0):
    """
    isb     : cardinal of the sub-block inside the memory (or root) block
    iLevel  : level of bipartition
    """
    ibb       = isb % 8
    scoord_r  = (ibb % (2**1))/(2**0)
    scoord_ph = (ibb % (2**2))/(2**1)
    scoord_th = (ibb % (2**3))/(2**2)
    """
    Depending on the value of iLevel, the 3-tuple that get_subcoord(..) 
    returns means different things. For instance,
    iLevel=0: the finest-grained location of the sub-block number isb.
    iLevel=1: the location of the octree in units of number of octrees.
    iLevel=2: the location of the octree that contains 8 octrees.
    ... so on ...
    """
    if iLevel>0:
        _i0, _i1, _i2 = get_subcoord(isb/8, iLevel=iLevel-1)
        scoord_r  += _i0*2
        scoord_ph += _i1*2
        scoord_th += _i2*2

    return scoord_r, scoord_ph, scoord_th

def deduce_terna(ib, nRoot=[8,8,4], nc=[6,4,4], nLevel=1):
    """
    According to documentation, X,Y,Z goes in the same order
    as R,LON,LAT.
    """
    nRootX, nRootY, nRootZ  = nRoot
    nc_r, nc_ph, nc_th      = nc
    # location inside the mem-blk, in units of sub-blocks
    # TODO: chekar si hay q reemplazar con 'nLevel'
    # TODO: probar con otros valores de nLevel!=1
    npart_D = 2**nLevel # number of bi-partitions in EACH DIMENSION
    ibb_r, ibb_ph, ibb_th = \
        get_subcoord(isb=ib%(npart_D**3), # cardinal inside the root block
            iLevel=nLevel)

    # the "octree" is composed by 2**(3*nLevel) children-blocks.
    ib_mem      = int(ib/(npart_D**3))
    ib_mem_r    = ib_mem % nRootX
    ib_mem_ph   = (ib_mem/nRootX) % nRootY
    ib_mem_th   = ib_mem / (nRootX*nRootY)

    return {
        'ib_mem':   [ib_mem_r, ib_mem_ph, ib_mem_th], 
        'ibb':      [ibb_r, ibb_ph, ibb_th],
        }


def make_sphere_shells(dir_src, dir_out, prefix_fig, ro, pazim=-60., clim=[None,None], verbose='debug', vnames=[], data_processor=None):
    prefix_inp = '3d__var_1_n'
    fnm_s = glob.glob(dir_src + '/' + prefix_inp + '*.out')
    # get the time labels as 'int' variables && sort it
    it_s = [ int(fnm.split('.out')[-2].split(prefix_inp)[-1]) for fnm in fnm_s ]
    it_s.sort()
    # number of digits in the time label (take the 1st element as sample)
    #ndigits = len(fnm_s[0].split('.out')[-2].split(prefix_inp)[-1])

    for it in it_s:
        fname_inp = dir_src + '/' + prefix_inp + '%08d.out'%it
        fname_fig = dir_out + '/' + prefix_fig + '%08d.png'%it
    
        logger.info(' [+] generating figure %d/%d ...\n' % (it, len(it_s)))
        _o = plot_sphere_cuts(fname_inp, fname_fig, ro, pazim, clim, 
            checks=False, complete=True, verbose=verbose, vnames=vnames,
            data_processor=data_processor)
        logger.info(' [+] saved: %s\n' % fname_fig)
        del _o


def plot_sphere_cuts(fname_inp, fname_fig, ro, pazim=-60., clim=[None,None], checks=False, complete=True, verbose='debug', vnames=[], data_processor=None, nc=None, nRoot=None, nLevel=None):
    """
    make 3D plot with a radial and longitudinal cuts
    """
    logger.setLevel(getattr(logging, verbose.upper()))
    r2d         = 180./np.pi
    #d           = get_array_Bmod(fname_inp, nc, nRoot, nLevel)
    assert len(vnames)>0, ' [-] We need names in vnames!\n'
    # we'll obtain:
    # - same data but in structured way; friendly for plot_surface().
    # - fill 'vdict'' with original ASCII data
    d           = get_array_vars(fname_inp, checks=checks, complete_domain_walk=complete, vnames=vnames, data_processor=data_processor)
    # NOTE: d['data'] is processed sutff built from the original (from
    # the ASCII file) simulation data. Such processing was made
    # by 'data_processor()'.
    # NOTE: a this point, 'vdict' has the original data from the ASCII file.
    r, ph, th   = d['coords']
    Bmod        = d['data']; 
    print ' [+] global extremes:', np.nanmin(Bmod), np.nanmax(Bmod)

    #--- slice an specific shell r=ro
    i_r = get_index_r(r, ro)
    print ' > We\'ll plot i_r: ', i_r

    # we need the transpose in order to be consistent with 'plot_surface'
    var_bare = Bmod.transpose((1,0,2))[:,i_r,:]
    # same w/o NaNs
    var, ph_clean, th_clean = clean_sparse_array(var_bare, ph, th)
    print '[+] plot extremes: ', np.nanmin(var), np.nanmax(var)
    # NOTE: 'plot_surface' can only plot variables with shape (n,m), so 
    # no 3D variables.
    cbmin, cbmax = [np.nanmin(var), np.nanmax(var)] if clim==[None,None] else clim
    print " >> ", np.nanmean(var), np.nanmedian(var)

    # mesh versions of the coords
    R, PH, TH   = np.meshgrid(r, ph_clean, th_clean)
    # get the cartesian coords (I know!)
    X           = R * np.cos(TH) * np.cos(PH)
    Y           = R * np.cos(TH) * np.sin(PH)
    Z           = R * np.sin(TH)

    #--- figure
    fig     = figure(1,)
    ax      = fig.add_subplot(111, projection='3d')
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
    'facecolors'    : cm.jet(norm(var)),
    #'edgecolors'    : 'none',
    }
    print '\n [*] Generating 3D plot...\n'
    surf = ax.plot_surface(X[:,i_r,:], Y[:,i_r,:], Z[:,i_r,:], **opt)
    # Note the cm.jet(..) --> cm.jet(norm(..)); see:
    # https://stackoverflow.com/questions/25023075/normalizing-colormap-used-by-facecolors-in-matplotlib

    # perspective azimuth
    ax.azim = pazim
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
    fig.savefig(fname_fig, dpi=135, bbox_inches='tight')
    close(fig)

    return d

#@profile
def PlotCut_fixed_ph(fig_stuff, data, pho, r_range, pazim=-60., verbose='debug'):
    """
    make 3D plot with a radial and longitudinal cuts
    """
    r2d         = 180./np.pi
    Bmod        = data['data']      # TODO: change ¡Bmod¡ to data/var/something
    r, ph, th   = data['coords']

    #--- slice an specific longitude slice at 'pho'
    i_ph = get_index_r(ph, pho/r2d)
    print ' > We\'ll plot i_ph: ', i_ph
    # set the plot range in 'r'
    i_r_min = get_index_r(r, r_range[0])
    i_r_max = get_index_r(r, r_range[1])

    # we'll search a interval in phi such that var_bare has some numeric values.
    # NOTE: 10 iterations seems reasonable.
    for i_dph in range(0,10):
        # we need the transpose in order to be consistent with 'plot_surface'
        var_bare = np.nanmean(
            Bmod.transpose((1,0,2))[i_ph-i_dph:i_ph+i_dph+1,i_r_min:i_r_max+1,:],
            axis = 0,
            )
        # if it has some numeric content, we have valid
        # data in 'var_bare', so we are done.
        if not np.isnan(np.nanmean(var_bare)): break

    # same w/o NaNs
    var, r_clean, th_clean = clean_sparse_array(var_bare,r[i_r_min:i_r_max+1],th)

    print '[+] plot extremes: ', np.nanmin(var), np.nanmax(var)
    # NOTE: 'plot_surface' can only plot variables with shape (n,m), so 
    # no 3D variables.
    print " >> mean, median: ", np.nanmean(var), np.nanmedian(var)

    # mesh versions of the coords
    R, TH  = np.meshgrid(r_clean, th_clean)
    # get the cartesian coords (I know!)
    X       = R * np.cos(TH) * np.cos(ph[i_ph])
    Y       = R * np.cos(TH) * np.sin(ph[i_ph])
    Z       = R * np.sin(TH)
    #--- figure
    fig_stuff['fig'], fig_stuff['ax'], surf = plot_stuff(
        [fig_stuff['fig'], fig_stuff['ax']], 
        coords = [X,Y,Z], 
        var = var.T, 
        norm = fig_stuff['norm'],
    )
    return {
    'FigAx'     : (fig_stuff['fig'], fig_stuff['ax']),
    'ph_plot'   : ph[i_ph],
    'surf'      : surf,
    }

#@profile
def PlotCut_fixed_r(fig_stuff, data, ro, pazim=-60., verbose='debug'):
    """
    make 3D plot with a radial and longitudinal cuts
    """
    r2d         = 180./np.pi
    # kws
    Bmod        = data['data']      # TODO: change ¡Bmod¡ to data/var/something
    r, ph, th   = data['coords']

    #--- slice an specific shell r=ro
    i_r = get_index_r(r, ro)
    print ' > We\'ll plot i_r: ', i_r

    # we need the transpose in order to be consistent with 'plot_surface'
    var_bare = Bmod.transpose((1,0,2))[:,i_r,:]
    # same w/o NaNs
    var, ph_clean, th_clean = clean_sparse_array(var_bare, ph, th)
    print '[+] plot extremes: ', np.nanmin(var), np.nanmax(var)
    # NOTE: 'plot_surface' can only plot variables with shape (n,m), so 
    # no 3D variables.
    #cbmin, cbmax = [np.nanmin(var), np.nanmax(var)] if clim==[None,None] else clim
    print " >> ", np.nanmean(var), np.nanmedian(var)

    # mesh versions of the coords
    PH, TH  = np.meshgrid(ph_clean, th_clean)
    # get the cartesian coords (I know!)
    X       = r[i_r] * np.cos(TH) * np.cos(PH)
    Y       = r[i_r] * np.cos(TH) * np.sin(PH)
    Z       = r[i_r] * np.sin(TH)
    #norm    = LogNorm(cbmin,cbmax) if kws.get('cscale','log')=='log' else Normalize(cbmin,cbmax)
    #--- figure
    fig_stuff['fig'], fig_stuff['ax'], surf = plot_stuff(
        [fig_stuff['fig'], fig_stuff['ax']], 
        coords = [X,Y,Z], 
        var = var.T, 
        norm = fig_stuff['norm'],
    )
    return {
    'FigAx'     : (fig_stuff['fig'], fig_stuff['ax']),
    'r_plot'    : r[i_r],
    'surf'      : surf,
    }


def plot_stuff(FigAx, coords, var, norm):
    fig, ax = FigAx
    X, Y, Z = coords
    #--- figure
    # other options
    opt = {
    'rstride'       : 1,
    'cstride'       : 1,
    'linewidth'     : 0,
    'antialiased'   : False,
    'shade'         : False,
    'alpha'         : 1., #kargs.get('alpha',0.9),
    'cmap'          : cm.jet,                # gray-scale
    'norm'          : norm,
    'vmin'          : norm.vmin, #kargs.get('cbmin',1),
    'vmax'          : norm.vmax, #kargs.get('cbmax',1000),
    'facecolors'    : cm.jet(norm(var)),
    #'edgecolors'    : 'none',
    }
    print '\n [*] Generating 3D plot...\n'
    surf = ax.plot_surface(X[:,:], Y[:,:], Z[:,:], **opt)
    # Note the cm.jet(..) --> cm.jet(norm(..)); see:
    # https://stackoverflow.com/questions/25023075/normalizing-colormap-used-by-facecolors-in-matplotlib
    return fig, ax, surf


def file_format(fname):
    cmd = 'file %s | awk -F: \'{print $2}\'' % fname
    p   = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT, bufsize=1, shell=True)
    std = p.communicate()
    assert p.returncode == 0, \
        '\n [-] something went wrong: %r\n' % std
    stdout = std[0]
    if stdout.startswith(' ASCII text'):
        return 'ascii'
    elif stdout.startswith(' Hierarchical Data Format (version 5) data'):
        return 'hdf5'
    elif stdout.startswith(' data'):
        return 'binary'
    else: 
        return None

#@profile
def make_3dplot(fname_inp, fname_fig, clim=[None,None], vnames=[], data_processor=None, verbose='debug', **kws):
    """
    make 3D plot with a radial and longitudinal cuts
    """
    logger.setLevel(getattr(logging, verbose.upper()))
    assert len(vnames)>0, ' [-] We need names in vnames!\n'
    # we'll obtain:
    # - same data but in structured way; friendly for plot_surface().
    # - fill 'vdict'' with original ASCII data
    d = get_array_vars(fname_inp, checks=False, complete_domain_walk=True, vnames=vnames, data_processor=data_processor)
    # NOTE: d['data'] is processed sutff built from the original (from
    # the ASCII file) simulation data. Such processing was made
    # by 'data_processor()'.
    # NOTE: a this point, 'vdict' has the original data from the ASCII file.
    r, ph, th   = d['coords']
    Bmod        = d['data']; 
    print ' [+] global extremes:', np.nanmin(Bmod), np.nanmax(Bmod)

    cbmin, cbmax = clim if clim is not [None,None] else (np.nanmin(Bmod),np.nanmax(Bmod))
    #--- figure
    fig_stuff   = {
    'fig'   : figure(1,),
    }
    fig_stuff.update({
    'ax'    : fig_stuff['fig'].add_subplot(111, projection='3d'),
    'norm'  : LogNorm(cbmin,cbmax) if kws.get('cscale','log')=='log' else Normalize(cbmin,cbmax),
    })
    fig     = fig_stuff['fig']
    ax      = fig_stuff['ax']
    norm    = fig_stuff['norm']

    #--- plot for fixed "r"
    o__fixed_r = PlotCut_fixed_r(fig_stuff, d, 
        ro = kws.get('ro', 5.0),
        pazim = kws.get('pazim',-60.),
        verbose = verbose,
    )
    fig, ax = o__fixed_r['FigAx']
    r_plot  = o__fixed_r['r_plot']
    surf_r  = o__fixed_r['surf']

    #--- plot for fixed "ph"
    r_range = kws.get('r_range', [1.0,7.0])
    pho     = kws.get('pho', 10.0)
    o__fixed_r = PlotCut_fixed_ph(fig_stuff, d, 
        pho = pho,
        r_range=r_range,
        pazim = kws.get('pazim',-60.),
        verbose = verbose,
    )
    fig, ax = o__fixed_r['FigAx']
    ph_plot = o__fixed_r['ph_plot']
    surf_ph = o__fixed_r['surf']

    # uniform axis limits
    axmin = np.min([getattr(ax,'get_%slim'%dim)() for dim in ('x','y','z')])
    axmax = np.max([getattr(ax,'get_%slim'%dim)() for dim in ('x','y','z')])
    ax.set_xlim(axmin,axmax)
    ax.set_ylim(axmin,axmax)
    ax.set_zlim(axmin,axmax)

    # perspective azimuth
    ax.azim = kws.get('pazim', -60.)
    sm = cm.ScalarMappable(cmap=surf_r.cmap, norm=fig_stuff['norm'])
    sm.set_array(d['data']); #surf.set_array(var)

    # labels && title
    ax.set_xlabel('X [Ro]')
    ax.set_ylabel('Y [Ro]')
    ax.set_zlabel('Z [Ro]')
    TITLE = '$r_o$ = %.2g $R_o$' % r_plot +\
    '\n($\phi_o$,r1,r2) : ($%g^o,%g\,Ro,%g\,Ro$)' % (pho,r_range[0],r_range[1])
    if kws.get('wtimelabel',False):
        tlabel = fname_inp.split('/')[-1].split('.h5')[0].split('_')[-1].replace('n','')
        TITLE += '\n step: '+tlabel
    ax.set_title(TITLE)

    #--- colorbar
    cb_label = '|B| [G]'
    cb_fontsize = 13
    axcb = fig.colorbar(sm, ax=ax)
    axcb.set_label(cb_label, fontsize=cb_fontsize)
    sm.set_clim(vmin=cbmin, vmax=cbmax)

    # save figure
    #show()
    fig.savefig(fname_fig, dpi=kws.get('dpi',135), bbox_inches='tight')
    close(fig)
    del fig
    return None


def lon_cut(fname_inp, fname_fig, lon=0.0, dlon=0.0, r_range=[1.,24.], clim=[None,None], verbose='debug', vnames=[], data_processor=None, cscale='linear', interactive=False):
    """
    make 2D plot with a longitudinal cuts
    """
    logger.setLevel(getattr(logging, verbose.upper()))
    r2d         = 180./np.pi
    assert len(vnames)>0, ' [-] We need names in vnames!\n'
    # we'll obtain:
    # - same data but in structured way; friendly for plot_surface().
    # - fill 'vdict'' with original ASCII data
    d = get_array_vars(fname_inp, checks=False, complete_domain_walk=True, vnames=vnames, data_processor=data_processor)
    assert d is not None
    # NOTE: d['data'] is processed sutff built from the original (from
    # the ASCII file) simulation data. Such processing was made
    # by 'data_processor()'.
    # NOTE: a this point, 'vdict' has the original data from the ASCII file.
    r, ph, th   = d['coords']   # shape (:,:,:)
    Bmod        = d['data']; 
    print ' [+] global extremes:', np.nanmin(Bmod), np.nanmax(Bmod)

    #--- slice an specific longitude = `lon`
    #i_ph  = get_index_r(ph*r2d, lon)
    #print ' > We\'ll plot i_ph: ', i_ph
    # set the plot range in 'r'
    i_r_min = get_index_r(r, r_range[0])
    i_r_max = get_index_r(r, r_range[1])
    # make selection in a given width in longitude (centered in
    # the `lon` value)
    cc_ph = (ph*r2d>=(lon-.5*dlon)) & (ph*r2d<=(lon+.5*dlon))
    if (cc_ph.nonzero()[0].size > 0) and (i_r_max - i_r_min + 1 > 0):
        print(' [+] phi plot limits: (%g, %g)' % (ph[cc_ph][0]*r2d,ph[cc_ph][-1]*r2d))
        print(' [+] r plot limits: (%g, %g)\n' % (r[i_r_min], r[i_r_max]))
    else:
        raise SystemExit('\n [-] the selection in longitude and radii is NULL!\n')
    
    logger.debug(' [+] averaging %d slices of phi ...'%len(cc_ph.nonzero()[0]))
    var_bare = np.nanmean(Bmod.transpose((1,0,2))[cc_ph,i_r_min:i_r_max+1,:], axis=0)
    # same w/o NaNs columns/rows
    var, r_clean, th_clean = clean_sparse_array(var_bare, r[i_r_min:i_r_max+1], th)
    #var_m = np.ma.masked_where(np.isnan(var),var)
    #var_m = np.ma.array(var, mask=np.isnan(var))
    var_m = np.ma.masked_where(np.isnan(var), var)

    print '[+] plot extremes: ', np.nanmin(var), np.nanmax(var)
    # NOTE: 'plot_surface' can only plot variables with shape (n,m), so 
    # no 3D variables.
    cbmin, cbmax = [np.nanmin(var), np.nanmax(var)] if clim==[None,None] else clim
    print " >> ", np.nanmean(var), np.nanmedian(var)

    # mesh versions of the coords
    R, TH   = np.meshgrid(r_clean, th_clean)
    # get the cartesian coords (I know!)
    RHO         = R * np.cos(TH)
    Z           = R * np.sin(TH)

    #--- figure
    fig     = figure(1, figsize=(6,5))
    ax      = fig.add_subplot(111, )

    #--- other options
    # color scale
    if cscale=='linear':
        norm = Normalize(cbmin, cbmax)
    elif cscale=='log':
        norm = LogNorm(cbmin, cbmax)
    else:
        raise SystemExit(' [-] invalid color scale: '+cscale+'\n')
    opt = {
    #'rstride'       : 1,
    #'cstride'       : 1,
    'linewidth'     : 0,
    #'antialiased'   : False,
    #'shade'         : False,
    #'shading'       : 'flat',
    #'alpha'         : 1., #kargs.get('alpha',0.9),
    'cmap'          : cm.jet,                # gray-scale
    'norm'          : norm,
    'vmin'          : cbmin, #kargs.get('cbmin',1),
    'vmax'          : cbmax, #kargs.get('cbmax',1000),
    'facecolors'    : cm.jet(norm(var_m)),
    #'interpolation' : 'none',
    'edgecolors'    : 'None',
    #'corner_mask'   : True,
    }
    print '\n [*] Generating 3D plot...\n'
    #surf = ax.contourf(RHO[:,:], Z[:,:], var_m[:,:].T, **opt)
    #surf = ax.pcolormesh(RHO[:,:], Z[:,:], var_m[:,:].T, **opt)
    #surf = ax.scatter(th_clean, r_clean, c=var[:,:], **opt)
    _ir, _ith = np.argwhere(~np.isnan(var[:,:])).T
    _r, _th  = r_clean[_ir], th_clean[_ith]
    _x, _y   = _r*np.cos(_th), _r*np.sin(_th)
    surf = ax.scatter(_x, _y, c=var[_ir,_ith], **opt)
    # Note the cm.jet(..) --> cm.jet(norm(..)); see:
    # https://stackoverflow.com/questions/25023075/normalizing-colormap-used-by-facecolors-in-matplotlib

    # perspective azimuth
    sm = cm.ScalarMappable(cmap=surf.cmap, norm=surf.norm)
    sm.set_array(var_m); #surf.set_array(var)

    ax.set_xlabel('$\\rho$ [Ro]')
    ax.set_ylabel('$Z$ [Ro]')
    TITLE = ' global $\phi$ limits: (%g, %g) \n' % (ph[0]*r2d, ph[-1]*r2d) +\
    '$\phi$ interval for plot: (%.2g, %.2g) [deg]\n' % (ph[cc_ph][0]*r2d, ph[cc_ph][-1]*r2d) +\
    '$r$ interval: (%.2g, %.2g) [Ro]' % (r[i_r_min], r[i_r_max])
    ax.set_title(TITLE)
    #ax.set_xlim(1.7, 1.9)
    #ax.set_ylim(-0.1, 0.1)

    #--- colorbar
    cb_label = '|B| [G]'
    cb_fontsize = 13
    axcb = fig.colorbar(sm, ax=ax)
    axcb.set_label(cb_label, fontsize=cb_fontsize)
    sm.set_clim(vmin=cbmin, vmax=cbmax)

    # save figure
    if interactive:
        show()
    else:
        fig.savefig(fname_fig, dpi=135, bbox_inches='tight')
    close(fig)

    return d

#@profile
def clean_sparse_array(m, x, y):
    """
    remove all rows and columns full of zeros
    """
    ni, nj = m.shape
    clean_j = list(range(0,nj))
    logger.info(' [+] cleaning sparse array......')
    for j in range(nj):
        if all(np.isnan(m[:,j])):
            # remove the element with value 'j'
            logger.debug(' [+] clearing slice with j=%d'%j)
            clean_j.remove(j)
    # clean columns
    m1 = m[:,clean_j]
    y_clean = y[clean_j]

    # now let's clean the rows
    clean_i = list(range(0,ni))
    for i in range(ni):
        if all(np.isnan(m[i,:])):
            # remove the i-th element 
            logger.debug(' [+] clearing slice with i=%d'%i)
            clean_i.remove(i)
    # clean rows
    m2 = m1[clean_i,:]
    x_clean = x[clean_i]

    # clean version of 'm'
    return m2, x_clean, y_clean
            


def get_domains(coords, eps=0.005, checks=True, complete=False, nc=[6,4,4], nRoot=[8,8,4], nLevel=1):
    """
    checks      : if True, if checks that the file size is consistent
                  with a complete structure of cells and children-blocks.
    complete    : if True, is walks all the ASCII file entries, one by one.
    """
    r, ph, th               = coords    # original coords from SWMF's output
    ndata                   = r.size

    if checks:
        nc_r, nc_ph, nc_th      = nc
        nRootX, nRootY, nRootZ  = nRoot
        npart_D = 2**nLevel # number of bi-partitions in EACH DIMENSION
        # nmbr of blocks (or "sub-blocks")
        nb_r                    = npart_D*nRootX
        nb_ph                   = npart_D*nRootY
        nb_th                   = npart_D*nRootZ
        # at least, it should multiple of the size of the smallest sub-block
        assert ndata % (nc_r*nc_ph*nc_th) == 0

        # we are assuming our hypothesis about the number of 
        # entries is NOT necessarily true. So we are not checking 
        # this assert.
        #assert nc_r*nb_r * nc_ph*nb_ph * nc_th*nb_th == ndata

    _r, _ph, _th    = [0.,], [0.,], [0.,]
    if complete:
        assert not checks, ' [-] flag "checks" must be False!\n'
        # we walk all the entries one by one, disregarding the
        # parameters 'nc', 'nRoot', 'nLevel'
        ind = 0
        while ind < r.size:
            if (ind % 4096 == 0.0):
                logger.debug(' [+] ind: %d/%d' % (ind,r.size))
            if not any(np.abs(np.array(_r)  - r[ind])  < eps):
                _r.append(r[ind]); 
            if not any(np.abs(np.array(_ph) - ph[ind]) < eps):
                _ph.append(ph[ind])
            if not any(np.abs(np.array(_th) - th[ind]) < eps):
                _th.append(th[ind])
            ind += 1
    else:
        # we assume the ASCII file has the entire structure of
        # cells and children-block
        assert not(complete) and checks,\
            ' [-] the flags "complete" and "check" must be False and True!\n'
        ib  = 0 
        while ib*(nc_r*nc_th*nc_ph) < r.size:
            #print ' sub-block: ', ib
            ind = ib*(nc_r*nc_ph*nc_th);
            if not any(np.abs(np.array(_r)  - r[ind])  < eps):
                _r.append(r[ind]); 
            if not any(np.abs(np.array(_ph) - ph[ind]) < eps):
                _ph.append(ph[ind])
            if not any(np.abs(np.array(_th) - th[ind]) < eps):
                _th.append(th[ind])
            ib += 1


    # delete the 1st element (= 0.0)
    _r  = np.array(_r)[1:]
    _ph = np.array(_ph)[1:]
    _th = np.array(_th)[1:]
    # sort them
    _r.sort(); _ph.sort(); _th.sort()

    if checks:
        logger.info(' [+] Check for size consistencies...')
        expect_r  = nb_r *nc_r  if complete else nb_r
        expect_ph = nc_ph*nc_ph if complete else nb_ph
        expect_th = nc_th*nc_th if complete else nb_th
        # check we did things right!
        assert _r.size  == expect_r, \
            '_r.size:%d; expected:%d'%(_r.size, expect_r)
        assert _ph.size == expect_ph
        assert _th.size == expect_th

    return _r, _ph, _th


def show_walk(fname, nc, nRoot, nLevel, checks=True, complete_domain_walk=False, prefix='walk', FirstLast=(None,None), dpi=100):
    np.set_printoptions(precision=2, linewidth=200)
    r, ph, th, Bmod         = read_data(fname)
    _r, _ph, _th = get_domains([r,ph,th], nc, nRoot, nLevel, 
        complete=complete_domain_walk, checks=checks)
    nc_r, nc_ph, nc_th  = nc
    eps = 0.005

    # fig
    fig     = figure(1, figsize=(12,8))
    ax      = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('R')
    ax.set_ylabel('PHI')
    ax.set_zlabel('THETA')
    # plot the Root blocks
    _opt = {'c':'b', 'marker':'^', 'alpha':0.4, 's':35}
    ax.scatter(        0,         0,         0, **_opt)
    ax.scatter(        0,         0, 2**nLevel-1, **_opt)
    ax.scatter(        0, 2**nLevel-1,         0, **_opt)
    ax.scatter(2**nLevel-1,          0,        0, **_opt)
    ax.scatter(2**nLevel-1, 2**nLevel-1,        0, **_opt)
    ax.scatter(        0, 2**nLevel-1, 2**nLevel-1, **_opt)
    ax.scatter(2**nLevel-1,         0, 2**nLevel-1, **_opt)
    ax.scatter(2**nLevel-1, 2**nLevel-1, 2**nLevel-1, **_opt)

    #while ib*(nc_r*nc_th*nc_ph) < r.size:
    #while ib < 240:
    # start block and final block
    nc_cell     = nc_r*nc_ph*nc_th
    assert FirstLast != (None,None), ' wrong input for first/last blocks!\n'
    ok_fl       = None not in FirstLast
    if None not in FirstLast:
        # plot in the range 'FirstLast'
        ib_ini, ib_end = FirstLast
    elif FirstLast[1] is not None:
        # plot the last FirstLast[1] blocks
        ib_ini  = r.size/nc_cell - FirstLast[1] #8*8*8 #0
        ib_end  = r.size/nc_cell - 1 #239
    elif FirstLast[0] is not None:
        # plot the first FirstLast[0] blocks
        ib_ini  = 0
        ib_end  = FirstLast[0] - 1

    # limits for 3D plot
    all__i_r  = [(np.abs(_r  -  r[_ib*nc_cell]) < eps).nonzero()[0][0] \
        for _ib in range(ib_ini,ib_end)]
    all__i_ph = [(np.abs(_ph - ph[_ib*nc_cell]) < eps).nonzero()[0][0] \
        for _ib in range(ib_ini,ib_end)]
    all__i_th = [(np.abs(_th - th[_ib*nc_cell]) < eps).nonzero()[0][0] \
        for _ib in range(ib_ini,ib_end)]
    ax.set_xlim(np.min(all__i_r),  np.max(all__i_r))
    ax.set_ylim(np.min(all__i_ph), np.max(all__i_ph))
    ax.set_zlim(np.min(all__i_th), np.max(all__i_th))

    # We'll walk the 1st point of every children-block (i.e. the 
    # smallest group of cells)
    for ib in range(ib_ini, ib_end+1):
        print ' sub-block (#%d): '%(ib-ib_ini), ib, '; ',
        ind = ib*(nc_r*nc_ph*nc_th);
        # find the coordinate where it coincides with any 
        # of the (_r, _ph, _th)
        i_r     = (np.abs(_r  -  r[ind]) < eps).nonzero()[0][0]
        i_ph    = (np.abs(_ph - ph[ind]) < eps).nonzero()[0][0]
        i_th    = (np.abs(_th - th[ind]) < eps).nonzero()[0][0]
        #if any(r > _r[-1]):
        #    import pdb; pdb.set_trace()
        print i_r, i_ph, i_th,
        print ';  %.2f %.2f %.2f' % (r[ind], ph[ind]*r2d, th[ind]*r2d)
        ax.scatter(i_r, i_ph, i_th, c='r', marker='o', s=5)
        ax_text = ax.text(8, 0, 18, 
            '(%d,%d,%d)'%(i_r,i_ph,i_th),
            fontsize=20)
        # remove the text from figure:
        # https://stackoverflow.com/questions/4981815/how-to-remove-lines-in-a-matplotlib-plot#13575495
        fname_fig = prefix + '_%05d'%(ib-ib_ini) + '.png'
        fig.savefig(fname_fig, dpi=dpi, bbox_inches='tight')
        ax_text.remove()

    #fig.show()
    close(fig)



if __name__=='__main__':
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
    '-v', '--verbose',
    type=str,
    default='debug',
    help='verbosity level (debug=minimal, info=extended)',
    )
    parser.add_argument(
    '-c', '--checks',
    action='store_true',
    default=False,
    help='checks size consitencies in the number of entries of the input file',
    )
    parser.add_argument(
    '-p', '--prefix',
    type=str,
    default='walk',
    help='prefix for figures',
    )
    parser.add_argument(
    '-fi', '--fname_inp',
    type=str,
    default='../../run__chip0_xxxvii/SC/IO2/3d__var_1_n00000005.out',
    help='input ASCII filename',
    )
    parser.add_argument(
    '-nLevel', '--nLevel',
    type=int,
    default=3,
    help='nLevel parameter of #GRIDLEVEL',
    )
    parser.add_argument(
    '-first', '--first',
    type=int,
    default=None,
    help='first ID of the sub-block',
    )
    parser.add_argument(
    '-last', '--last',
    type=int,
    default=None,
    help='last ID of the sub-block',
    )
    parser.add_argument(
    '-dpi', '--dpi',
    type=int,
    default=100,
    help='dpi parameter for pylab.figure.savefig()',
    )
    pa = parser.parse_args()
    #--- numpy print settings
    np.set_printoptions(precision=2, linewidth=230)
    #--- logging
    if pa.verbose in ('debug', 'info'):
        logger.setLevel(getattr(logging, pa.verbose.upper()))
    else:
        raise SystemExit(' [-] Invalid argument: %s\n'%pa.verbose)

    #--- 
    #_r, _ph, _th = get_domains(fnm, [6,4,4], [8,8,4], 2)
    #for isb in range(60):
    #    i_r, i_ph, i_th = get_subcoord(isb, iLevel=2)
    #    print isb, i_r, i_ph, i_th
    
    #--- walking
    #try:
    #    show_walk(pa.fname_inp, [6,4,4], [8,8,4], pa.nLevel, checks=pa.checks, complete_domain_walk=False, prefix=pa.prefix, FirstLast=(pa.first,pa.last), dpi=pa.dpi)
    #except KeyboardInterrupt:
    #    print " > Keyboard Interrupt... "

    #--- new extraction method
    o = get_array_vars(pa.fname_inp, 
            checks=pa.checks,
            complete_domain_walk=True,
            vdict=None)

#EOF
