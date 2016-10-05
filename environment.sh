#!/bin/bash

module load python/2.7-anaconda

if [ ${NERSC_HOST} == "edison" ]; then
module load root/6.06.04
else
module load root/6.06.06
fi

#some special treatment required here
#if [ ${NERSC_HOST} == "gerty" ]; then
#
#export ROOT_DIR=$(module show root/6.06.06 2>&1 > /dev/null | grep ROOT_DIR | awk '{print $3}' | sed 's|/usr/common/software/|/global/common/cori/software/|g')
#
#export PATH=$(module show root/6.06.06 2>&1 > /dev/null | grep PATH | grep -v LD_LIBRARY_PATH | awk '{print $3}' | sed 's|/usr/common/software/|/global/common/cori/software/|g'):${PATH}
#
#export LIBRARY_PATH=$(module show root/6.06.06 2>&1 > /dev/null | grep LIBRARY_PATH | grep -v LD_LIBRARY_PATH | awk '{print $3}' | sed 's|/usr/common/software/|/global/common/cori/software/|g'):${LIBRARY_PATH}
#
#export LD_LIBRARY_PATH=$(module show root/6.06.06 2>&1 > /dev/null | grep LD_LIBRARY_PATH | awk '{print $3}' | sed 's|/usr/common/software/|/global/common/cori/software/|g'):${LD_LIBRARY_PATH}
#
#export PYTHONPATH=$(module show root/6.06.06 2>&1 > /dev/null | grep PYTHONPATH | awk '{print $3}' | sed 's|/usr/common/software/|/global/common/cori/software/|g'):${PYTHONPATH}
#fi
