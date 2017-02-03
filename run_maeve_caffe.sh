
export THEANO_FLAGS=device=gpu$1
shift
python atlas_main_caffe.py $@
