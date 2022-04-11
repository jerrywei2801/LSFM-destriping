#============================================================
#
#   Script to
#   - Execute the testing
#
#============================================================

import os, sys


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

run_GPU = '' if sys.platform == 'win32' else ' THEANO_FLAGS=device=gpu,floatX=float32 '
print("\nRun the prediction on GPU ")
os.system(run_GPU +' python ./testing.py')
