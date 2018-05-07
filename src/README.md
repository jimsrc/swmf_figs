<!--- BOF -->
## Examples

---
### generate "animation" (series of .png files)
Generate animation of the time evolution of a given scaler observable (i.e. temp, |B|, density, etc).
For example:

    ./make_anim.sh \
        -dir-dst ./figs__CR2106/ro_2.0__rr_2-20 \
        -fprefix ~/swmf_chip1/run__CR2106_FDIPS_hdf5/SC/IO2/3d__var_1_n \
        -vname temp \
        -ro 2.0 \
        -rr 2.0 20.0 \
        -clim 145e4 156e

**IMPORTANT**: If you pass the last two arguments in a different order, they are 
not passed correctly to the python script! :S (don't know WHY).
where:
* ´-dir-dst´: destination directory
* ´-fprefix´: full-path prefix of the input files
* ´-vname´	: name of the scalar observable
* ´-clim´	: limits (two values) for the colorbar. If not used, the limits are set automatically in each frame.


---
### convert from ASCII to HDF5 (massive files):
Change the path in variable `dir_src` (input directory, where the 
ASCII files are), and run:

    ./convert_to_h5.sh
    # this bash script calls the tt.py script, which converts 
    # one ASCII file at a time.


---
### generate 3D plots (massive files):



---
### show a 3D plot with mixed cuts (individual files):
* cut in r (i.e. a spheric shell), determinated by the argument `-ro`.
* cut in `ph`, in a given range of radii (`r1`:`r2`), determinated by the arguments `-pho` (longitude value, in degrees, at which we take a slice) and `-rr`.
NOTE: the input file can be in ASCII or HDF5 format.

```bash
./test.py -- \
    -fi ../../run__FDIPS_iv/SC/IO2/3d__var_1_n00000005.out \
    -ff ./figs/test_3.png \
    -cs log -vname Bmod -ro 5.0 \
    -pho 10.0 -rr 5.0 10.0 -clim 0.008 0.08
# NOTE: the colobar scale is determinated with the argument -cs
```

---
### cut at a given longitude phi

    ./single_lon_cut.py -- -v debug \
        -rr 1. 6. \
        -fi ../../run__FDIPS_iv/SC/IO2/3d__var_1_n00000005.out \
        -ff figs/lon_40deg.png \
        -lon 40. \
        -dlon 3. \
        -vname Bmod

    ./single_lon_cut.py --pdb -- -v debug -rr 1.01 2.3 -fi ~/swmf.lfs__FDIPS_CR2106/run_test/SC/IO2/3d__var_1_n00000000.out -ff ./test.png -lon 40. -dlon 3. -vname Bmod -i


---
### with any `nLevel`

    ./several_rcuts.py -- -v debug \
        -ro 5. \
        -ff ro_5.0__temp__FDIPS_iv.png \
        -fi ../../run__FDIPS_iv/SC/IO2/3d__var_1_n00000005.out \
        -vname temp

# massive generation of figures
./several_rcuts.py -- -v debug -ro 5. -ds ../../run__FDIPS_iv/SC/IO2 -vname temp
-ff ro_5.0__temp__FDIPS_iv__ -df ./figs -clim 1450000 1650000
```


---
### Test w/ nLevel=1
```bash
# get the output from that tag:
git show chip0_xxxiii:run_test/SC/IO2/3d__var_1_n00000005.out > 3d__var_1_n00000005__chip0_xxxiii.out

# now use it with nLevel=1
./several_rcuts.py --pdb -- --checks -ff test_00.png -fi ./3d__var_1_n00000005__chip0_xxxiii.out -nLevel 1
```

---
### Test w/ nLevel=2
This generates a 3D plot successfully:
```bash
# checking if the ASCII contents are consistent with a complete grid of cells and childre-blocks
./several_rcuts.py -- --checks -ff test_01.png -fi ../../run__chip0_xxxv/SC/IO2/3d__var_1_n00000005.out -nLevel 2

# w/o any checking, just read the bare contents
./several_rcuts.py -- -v debug -ff test_00__chip0_xxxv.png -fi ../../run__chip0_xxxv/SC/IO2/3d__var_1_n00000005.out
```

---
### Test w/ nLevel=3
```bash
# w/o any checking, just read the bare contents
./several_rcuts.py -- -v debug -ff test_00__chip0_xxxvii.png -fi ../../run__chip0_xxxvii/SC/IO2/3d__var_1_n00000005.out
```


---
### walk an ASCII file with unknown structure:
```bash
#-- nLevel=2
# between the 
./funcs.py -- -v info -fi ../../run__chip0_xxxv/SC/IO2/3d__var_1_n00000005.out -nLevel 2 -p ./figs/nLevel.2_first_ --first 512

#-- nLevel=3
# the 1st 512 children-blocks
./funcs.py -- -v info -fi ../../run__chip0_xxxvii/SC/IO2/3d__var_1_n00000005.out -nLevel 3 -p ./figs/nLevel.3_first_ --first 512
# between the 512 and 1024
./funcs.py -- -v info -fi ../../run__chip0_xxxvii/SC/IO2/3d__var_1_n00000005.out -nLevel 3 -p ./figs/nLevel.3_select_ --first 512 --last 1024
```

### Doesn't work for `nLevel=3`:
this:
```bash
./funcs.py -- -v info -fi ../../run__chip0_xxxvii/SC/IO2/3d__var_1_n00000005.out -nLevel 3 -p ./figs/nLevel.3_first2_ --first 512 --dpi 150
```
shows that the ASCII file walks the grid in a weird way! ¿???
NOTE: the `*.idl` ASCII files also walk the grid in the same way as the `*.out`.

<!--- EOF -->
