# These correspond to the fields of the 
# run in commit <FDIPS_iv>

#++++++++++++++++++++++++++++++++++++++++
# names of the fields in the IDL-ASCII/binary file
vnames = [
'bx', 'by', 'bz', 'rho', 'p', 'ux', 'uy', 'uz', 'te', 'ti', 'I01', 'I02',
]

# variables (corresponding to the process_..() functions below) that
# will be saved into the HDF5 files. This is necessary when converting 
# from ASCII to HDF5.
ovnames  = ['b', 'p', 'rho', 'u', 'te', 'ti', 'I01', 'I02']
# list of variables (included in 'ovnames') to be treates/saved as **vectors**
ovectors = ['b', 'u']

#++++++++++++++++++++++++++++++++++++++++
# methods that return scalars as function of 
# the original variables in the ASCII file
import numpy as np

def process_Bmod(vdict, **kw):
    try:
        bx, by, bz = vdict['bx'], vdict['by'], vdict['bz'] # when reading ASCII file
    except:
        bx, by, bz = vdict['b'].transpose((3,0,1,2)) # when reading the HDF5 file
    Bmod = np.sqrt(bx*bx + by*by + bz*bz)
    return Bmod

def process_Bvert(vdict, **kw):
    """
    Calculate the radial (or "vertical") component of the
    magnetic field.
    """
    try:
        bx, by, bz = vdict['bx'], vdict['by'], vdict['bz'] # when reading ASCII file
    except:
        bx, by, bz = vdict['b'].transpose((3,0,1,2)) # when reading the HDF5 file

    r, ph, th = kw['r'], kw['ph'], kw['th']

    # initialize b radial
    br = np.ones((r.size,ph.size,th.size),dtype=np.float32)*np.nan
    for ir in range(r.size):
        print 'ir: %d/%d' % (ir+1,r.size)
        for iph in range(ph.size):
            for ith in range(th.size):
                # if this is NaN, there's no need to calculate, since
                # 'br' has NaN values from initialization.
                if np.isnan(bx[ir,iph,ith]): continue

                # unitary vector of this position
                xu = np.cos(np.pi-th[ith]) * np.cos(ph[iph])
                yu = np.cos(np.pi-th[ith]) * np.sin(ph[iph])
                zu = np.sin(np.pi-th[ith])
                # b field at this position, in cartesian coords
                b  = [bx[ir,iph,ith], by[ir,iph,ith], bz[ir,iph,ith]]
                # radial component of b, at this position
                br[ir,iph,ith] = np.inner([xu,yu,zu], b)

    return br

    ## direction of the unitary vector of the position
    #x_ = np.cos(np.pi-th) * np.cos(ph)
    #y_ = np.cos(np.pi-th) * np.sin(ph)
    #z_ = np.sin(np.pi-th)
    ## the inner product of (x_,y_,z_) with b=(bx,by,bz) gives
    ## the radial component of b.
    #pos = np.array([x_,y_,z_]).T
    #b   = np.array([bx,by,bz]).T
    ## only the diagonal elements of this array are useful
    #Br_ = np.inner(pos,b)
    #Br  = [ Br_[i,i] for i in range(bx.size) ]


    #Bmod = np.sqrt(bx*bx + by*by + bz*bz)
    #return Bmod

def process_Umod(vdict, **kw):
    try:
        ux, uy, uz = vdict['ux'], vdict['uy'], vdict['uz'] # when reading ASCII file
    except:
        ux, uy, uz = vdict['u'].transpose((3,0,1,2)) # when reading the HDF5 file
    Umod = np.sqrt(ux*ux + uy*uy + uz*uz)
    return Umod

def process_b(vdict, **kw):
    bx, by, bz = vdict['bx'], vdict['by'], vdict['bz'] # when reading ASCII file
    return [bx,by,bz]

def process_u(vdict, **kw):
    ux, uy, uz = vdict['ux'], vdict['uy'], vdict['uz'] # when reading ASCII file
    return [ux,uy,uz]

def process_p(vdict, **kw):
    return vdict['p']

def process_n(vdict, **kw):
    return vdict['n']

def process_te(vdict, **kw):
    return vdict['te']

def process_ti(vdict, **kw):
    return vdict['ti']

def process_I01(vdict, **kw):
    return vdict['I01']

def process_I02(vdict, **kw):
    return vdict['I02']

def process_temp(vdict, **kw):
    return vdict['temp']

def process_rho(vdict, **kw):
    return vdict['rho']

#EOF
