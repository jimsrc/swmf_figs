#!/bin/bash

#--- grab args
[[ $# -eq 0 ]] && {
    # TODO: print help.
    echo -e "\n [-] we need arguments! XD\n";
    exit 1;
}
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -dir)           # input dir
        DirSrc=$2
        shift; shift
        ;;
        -nproc)         # number of processors
        NPROC=$2 
        shift; shift
        ;;
        -movie)         # just make the movie, don't build figs
        JustMovie=1 
        shift; shift
        ;;
        -ro)            # radial coordinate for the radial cut
        ro=$2
        shift; shift
        ;;
        *)
        echo "\n [-] unknown argument: $1\n"
        exit 1
    esac
done

# run in a SINGLE processor by default
NPROC=${NPROC:-1}

# i/o paths
if [[ "$DirSrc" == "" || ! -d $DirSrc ]]; then
    echo -e " [-] source dir is empty or doesn't exist:\n $DirSrc\n"
    exit 1;
fi

DirDst="${DirSrc}/rcut=${ro}"
mkdir -pv $DirDst

# activate conda env
source activate spacepy2 || exit 1

# scalar observables
VNAMEs=(rho p te ti Bmod Umod Bvert)
# labels for the colorbar
# NOTE: a quick list of the units in which the SWMF produces
# the output, is in p.69 of the SWMF.pdf manual.
CLABELs=('$\rho$ [amu/cc]' 'p [nPa]' 'pe [K]' 'ti [K]' 'Bmod [G]' 'Umod [km/s]' '$B_r$ [G]')

[[ $JustMovie -eq 0 ]] && for i in $(seq 0 $((${#VNAMEs[@]}-1))); do
    vname=${VNAMEs[$i]}
    clabel=${CLABELs[$i]}
    echo -e "\n [*] processing vname: $vname\n"
    mpirun -np $NPROC ./test.py -- r_cut \
        -ds ${DirSrc} \
        -dd ${DirDst} \
        -ro $ro \
        -figsize 8 4 \
        -cs linear \
        -vname $vname \
        -cm viridis \
        -cl "$clabel" &
    wait $! || {
        echo -e "\n [-] s'thing went wrong with vname: ${vname}\n";
        exit 1;
        }
done


#--- movies
for vname in ${VNAMEs[@]}; do
    pattern="3d__var_1_n*__${vname}.png"
    out=anim__${vname}__CR2106.mpg
    # generate movie .mpg
    docker run --rm -it \
        -v $DirDst:/data \
        lamp/mencoder:ubuntu14.04 \
            mencoder "mf:///data/$pattern" \
            -mf type=png:fps=2 \
            -ovc copy \
            -o /data/$out \
        || { echo -e "\n [-] Couldn't generate the .mpg!\n"; exit 1; } \
        && echo -e "\n [+] generated: $DirDst/$out\n"

    # convert .mpg --> .gif
    fname_gif=${out%.*}.gif
    docker run --rm -it \
        -v $DirDst:/data \
        mencoder/ubuntu16.04:latest \
        ffmpeg -i /data/$out -pix_fmt rgb24 -loop 0 /data/${fname_gif} \
    || { echo -e "\n [-] Couldn't convert $out to .gif !\n"; exit 1; } \
    && {
        echo -e "\n [+] MPEG version:\n $DirDst/${out}\n";
        echo -e "\n [+] GIF  version:\n $DirDst/${fname_gif}\n";
    }
done

#EOF
