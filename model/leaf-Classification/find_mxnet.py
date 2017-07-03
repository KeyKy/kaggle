import sys
import os

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../../3dparty/mxnet/python'))

main_path = os.path.join(curr_path, '../../')
sys.path.insert(0, main_path)

try:
    import mxnet as mx
    print(mx.__file__)
except:
    print('wrong path of mxnet')
