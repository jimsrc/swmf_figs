
# SWMF-POST

Tools to post-process output simulations from the SWMF framework (see 
[here](http://csem.engin.umich.edu/tools/swmf/) and 
[here](https://ccmc.gsfc.nasa.gov/models/modelinfo.php?model=BATS-R-US)).

---
## Overview

When dealing with the output data from SWMF simulations, we often need to deal with IDL binary/ASCII formats, which leads us to use IDL (i.e. propietary software).

This is an open-source/free-software alternative to post-process those binary files, using solely Python libraries.

This repository deals with:
* conversion of the IDL binaries/ASCII formats from SWMF to HDF5 format, stored in an array structure that is friendly with Matplotlib routines.
* 2D/3D plotting routines, making cuts in spherical coordinates. The previous step of getting the HDF5 versions of tha data saves a lot of time when running the plotting scripts.


---
For description of the tools see the sources [README](src/README.md).


<!--- EOF -->
