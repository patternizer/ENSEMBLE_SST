. /gws/nopw/j04/fiduceo/Users/mtaylor/anaconda3/etc/profile.d/conda.sh
export PATH=$PATH
conda activate mike
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1

#mkdir -p /dev/shm/mike/cache
#echo "Environment:"
#export
#echo "ulimit:"
#ulimit -a
#echo "conda"
#conda list
#python3.6 -X faulthandler $@

$@



