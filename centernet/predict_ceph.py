from cgi import test
import time

import numpy as np, pandas as pd
import tensorflow as tf
from PIL import Image

from centernet import CenterNet

import os, csv
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#dir_origin_path = "../../Centernet/ceph/test/"
dir_origin_path = "/data3/chang/ypark/Centernet/predict/inputs"
dir_save_path   = "/data3/chang/ypark/Centernet/predict/output"

Landmarks = ['Knee',
'Elbow',
'Hand',
'Ankle',
'Neck',
'Shoulder',
'Pelvis','Side']

'''Head','R_pelvis','L_pelvis','Navel',
                             'R_shoulder',
                             'Lower_neck',                            
                             'L_shoulder',                             
                             'R_hand',                            
                             'L_hand',                            
                             'R_knee',
                             'L_knee',                            
                             'R_elbow',                             
                             'L_elbow',
                             'R_ankle',
                             'L_ankle'''

o_cols = ['Knee_x','Knee_y',
'Elbow_x','Elbow_y',
'Hand_x','Hand_y',
'Ankle_x','Ankle_y'
'Neck_c','Neck_y',
'Shoulder_x','Sholuer_y',
'Pelvis_x','Pelvis_y','Side_x','Side_y']
    
'''Head_x','Head_y',                             
 'R_pelvis_x','R_pelvis_y',                             
 'L_pelvis_x','L_pelvis_y',                                                          
'Navel_x','Navel_y',
 'R_shoulder_x ','R_shoulder_y',
 'Lower_neck_x','Lower_neck_y',                            
 'L_shoulder_x ','L_shoulder_y',                           
 'R_hand_x','R_hand_y',                            
 'L_hand_x','L_hand_y',                            
 'R_knee_x','R_knee_y',
 'L_knee_x','L_knee_y',                            
 'R_elbow_x','R_elbow_y',                             
 'L_elbow_x','L_elbow_y',
 'R_ankle_x','R_ankle_y',
 'L_ankle_x','L_ankle_y'''


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def new_func(image_save, test_tf, centernet, image):
    (ret_dict, r_image) = centernet.detect_image_ceph(image, image_save=image_save, text=test_tf)
    return ret_dict,r_image

if __name__ == "__main__":

    mode = "dir_predict"
    
    
    image_save = False

    '''test_tf = input('Write score text on image? (T/F) : ')

    if test_tf=='T' or test_tf=='t':
        test_tf=True
    elif test_tf=='F' or test_tf=='f':
        test_tf=False'''
    test_tf=True


    image_save_tf = 't'
    if image_save_tf.lower()=='t':
        image_save = True


    # model
    centernet = CenterNet(heatmap = False)

    img_names = os.listdir(dir_origin_path)


    point_array = [] # celes

    #try:
    with open('/data3/chang/ypark/Centernet/centernet-tf2-main-20220719T121658Z-001/centernet-tf2-main/csv/out_test.csv', 'w', newline="") as f:

            f.write('num, ID,')
            for col in o_cols:
                f.write(f'{col},')
            f.write('\n')

            for subj_int_id, img_name in tqdm(enumerate(img_names)):

                if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    image_path  = os.path.join(dir_origin_path, img_name)
                    image       = Image.open(image_path)
                    subj_id, _ = os.path.splitext(os.path.basename(img_name))                    
                
                    ret_dict, r_image = new_func(image_save, test_tf, centernet, image)

                    
                    ########################
                    print(len(ret_dict))
                    print(ret_dict)

                
                    # -----------------------------
                    if image_save :
                        if not os.path.exists(dir_save_path):
                            os.makedirs(dir_save_path)
                        r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                    # -----------------------------
                
                    f.write(f'{subj_int_id}, {subj_id},')

                    for idx,l in enumerate(Landmarks):
                        idx = idx*2

                        #x, y = ret_dict.get(l, (np.nan,np.nan)) 
                        x, y = ret_dict.get(l, ('-1', '-1')) 
                        f.write(f'{y},{x},')
                    f.write('\n')
                
            f.close()

    #except IOError:
        #print('\n [Log] IOError')        