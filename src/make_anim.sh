#!/bin/bash
exe=./test.py
ext=h5

#++++++++++++++++++++++++++++++++++++++++++
# arggument parser
#++++++++++++++++++++++++++++++++++++++++++
while [[ $# -gt 0 ]]
do
key="$1"

# default values
clim_arg=""

case $key in
    -fprefix) # full-path prefix of the input files
    fprefix="$2"
    shift # past argument
    shift # past value
    ;;
    -dir-dst)  # directory for figures
    dir_dst="$2"
	shift # past argument
    shift # past argument
    ;;
    -vname) # name of the variable
    vname="$2"
	shift # past argument
    shift # past argument
    ;;
    -ro) # name of the variable
    ro=$2
	shift # past argument
    shift # past argument
    ;;
    -clim)  # directory for figures
	# NOTE1: we include apostrofesis symbol (") in case
	# any of the arguments if a signed number (i.e. includes a minus symbol)
	# NOTE2: the note above doesn't apply because it doesn't run if I include " symbol.
	# TODO: look for a way to allow signed numbers!
	clim_arg="-clim $2 $3"
	shift # past argument
    shift
    shift # past arguments
    ;;
    -rr)  # radii range
	rr_arg="-rr $2 $3"
	shift # past argument
    shift
    shift # past arguments
    ;;
    *)    # unknown option
	echo -e " [-] argument not recognized: $1\n" && exit 1
    ;;
esac
done
#++++++++++++++++++++++++++++++++++++++++++


# check directory for figures
if [[ ! -d ${dir_dst} ]]; then
    mkdir -p ${dir_dst}
fi

# check -vname
[[ "$vname" == "" ]] \
	&& {
		echo -e "\n [-] must specify a variable name with -vname\n" && exit 1
	}

# report
echo -e " [*] clim      : ${clim_arg}"
echo -e " [*] r-range   : ${rr_arg}"
echo -e " [*] ro        : $ro"
echo

_fs=$(ls $fprefix*.$ext)
for _f in $_fs; do
    # grab the basename && deduce a fname for the .png
    fpng=${dir_dst}/$(basename $_f | sed "s/\.$ext/__$vname\.png/g")

    # report
    echo -e " [*] input     : $_f"
    echo -e " [+] saving to : $fpng\n"

    # execute
    $exe -- -fi $_f -ff $fpng \
        -cs linear \
        -vname $vname \
        -ro $ro \
        -pho 10.0 \
        ${rr_arg} \
		${clim_arg} # empty string if not specified
        #-clim 145e4 156e4

    # exit if one of the iterations fail
    [[ $? -ne 0 ]] && exit 1
done
#EOF
