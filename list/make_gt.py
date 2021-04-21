import numpy as np
import os
import cv2

clip_len = 16

# the dir of testing images
video_root = 'xx/TestClips/Videos'   ## the path of test videos
feature_list = 'rgb_test.list'
# the ground truth txt

gt_txt = 'xx/annotations.txt'     ## the path of test annotations
gt_lines = list(open(gt_txt))
gt = []
lists = list(open(feature_list))
tlens = 0
vlens = 0
for idx in range(len(lists)):
    name = lists[idx].strip('\n').split('/')[-1]
    if '__0.npy' not in name:
        continue
    name = name[:-7]
    vname = name+'.mp4'
    cap = cv2.VideoCapture(os.path.join(video_root,vname))
    lens = int(cap.get(7))

    # the number of testing images in this sub-dir

    gt_vec = np.zeros(lens).astype(np.float32)
    if '_label_A' not in name:
        for gt_line in gt_lines:
            if name in gt_line:
                gt_content = gt_line.strip('\n').split()
                abnormal_fragment = [[int(gt_content[i]),int(gt_content[j])] for i in range(1,len(gt_content),2) \
                                        for j in range(2,len(gt_content),2) if j==i+1]
                if len(abnormal_fragment) != 0:
                    abnormal_fragment = np.array(abnormal_fragment)
                    for frag in abnormal_fragment:
                        gt_vec[frag[0]:frag[1]]=1.0
                break
    mod = (lens-1) % clip_len # minusing 1 is to align flow  rgb: minusing 1 when extracting features
    gt_vec = gt_vec[:-1]
    if mod:
        gt_vec = gt_vec[:-mod]
    gt.extend(gt_vec)
    if sum(gt_vec)/len(gt_vec):
        tlens += len(gt_vec)
        vlens += sum(gt_vec)

np.save('gt.npy', gt)