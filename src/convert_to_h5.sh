#!/bin/bash

dir_src=$HOME/swmf.lfs__FDIPS_CR2106/run_test/SC/IO2

for _f in $(ls $dir_src/*.out); do
    echo -e "\n +++++++++++++++++++++++++++++++++++++++++++++"
    echo -e " [*] [$0] src: $_f"
    f_h5=$(echo $_f | sed 's/\.out/\.h5/g')
    echo -e " [*] [$0] dst: ${f_h5}"
    ./convert_to_hdf5.py --pdb -- -fi ${_f} -fo ${f_h5}

    if [[ $? -ne 0 ]]; then
        echo -e "\n [-] something went wrong!\n"
        exit 1
    fi
done
