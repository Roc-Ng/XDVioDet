import numpy as np
import os
import glob

root_path = '/media/peng/Samsung_T51/vggish-features/test'
files = sorted(glob.glob(os.path.join(root_path, "*.npy")))
violents = []
normal = []
with open('audio2_test.list', 'w+') as f:
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)

# '''
# FOR UCF CRIME
# '''
# root_path = '/media/peng/Store/UCF_Crimes/c3d_features/train/RGB'
# dirs = os.listdir(root_path)
# print(dirs)
# with open('ucf-c3d.list','w+') as f:
#     normal = []
#     for dir in dirs:
#         files = sorted(glob.glob(os.path.join(root_path, dir, "*.npy")))
#         for file in files:
#             if 'Normal_' in file:
#                 normal.append(file)
#             else:
#                 newline = file+'\n'
#                 f.write(newline)
#     for file in normal:
#         newline = file+'\n'
#         f.write(newline)