import pandas as pd 
import numpy as np
import os
from xml.etree.ElementTree import Element, SubElement, ElementTree


# label_data : csv file

#,Neck_x,Neck_y,R_elbow_x,R_elbow_y,L_elbow_x,L_elbow_y,R_pelvis_x,R_pelvis_y,L_pelvis_x,L_pelvis_y,R_ankle_x,R_ankle_y,L_ankle_x,L_ankle_y,R_knee_x,R_knee_y,L_knee_x,L_knee_y,R_hand_x,R_hand_y,L_hand_x,L_hand_y,R_shoulder_x,R_shoulder_y,L_shoulder_x,L_shoulder_y,Navel_x,Navel_y,Image_name
#landmark_name = ['Glabella', 'R3', 'Nasion']
#,Knee_x,Knee_y,Elbow_x,Elbow_y,Hand_x,Hand_y,Ankle_x,Ankle_y,Neck_x,Neck_y,Shoulder_x,Shoulder_y,Pelvis_x,Pelvis_y,Side_x,Side_y
landmark_name = [

    'Neck',
'R_elbow',
'L_elbow',
'R_pelvis',
'L_pelvis',
'R_ankle',
'L_ankle',
'R_knee',
'L_knee',
'R_hand',
'L_hand',
'R_shoulder',
'L_shoulder',
'Navel']
    
                             #'Knee',
                             #'Elbow',
                             #'Hand',
                             #'Ankle',
                             #'Neck',
                             #'Shoulder',
                             #'Pelvis',
                             #'Side'
                            
#]



def xml_writer(label_data, dsVOC, year):
    
    len_file = len(label_data['ID'])

    bbox_size = 6
        
    for i in range(len_file):
        root = Element("annotation")

        element1 = Element("filename")
        
        root.append(element1)
       
        
        element1.text =  label_data.iloc[i]['ID']
        


        element2 = Element("size")
        root.append(element2)
        
        sub_element2 = SubElement(element2, "width")
        sub_element2.text = '256'
        sub_element2 = SubElement(element2, "height")
        sub_element2.text = '256'
        sub_element2 = SubElement(element2, "depth")
        sub_element2.text = '3'

        

        for j in range( len(landmark_name) ):
            element3 = Element("object")
            root.append(element3)
            sub_element3 = SubElement(element3, "name")
            sub_element3.text = landmark_name[j]
            
            x_loc = label_data.iloc[i][landmark_name[j]+'_x'].astype(np.float)
            y_loc = label_data.iloc[i][landmark_name[j]+'_y'].astype(np.float)
            
            
            tmp = np.array([(x_loc/4.21875)- bbox_size, (x_loc/4.21875) + bbox_size, (y_loc/7.5) - bbox_size, (y_loc/7.5) +bbox_size]).astype(int)
            #1080,1920/1.125,2 S
            # 512,512 /  2.2, 3.75
            # 360,360 / 5.34, 3
            #256,256 / 4.21875,7.5
###

            sub_element4 = SubElement(element3, "bndbox")

            sub_element5 = SubElement(sub_element4, "xmin")
            sub_element5.text = str(tmp[0])

            sub_element6 = SubElement(sub_element4, "ymin")
            sub_element6.text = str(tmp[2])

            sub_element7 = SubElement(sub_element4, "xmax")
            sub_element7.text = str(tmp[1])

            sub_element8 = SubElement(sub_element4, "ymax")
            sub_element8.text = str(tmp[3])
            
        
        tree = ElementTree(root)
        treef = [root]
        #i_2 = '{0:04d}'.format(i)
        i_2 = label_data['ID'][i]
        
        fileName = f"../centernet-tf2-main/VOCdevkit/{dsVOC}{year}/Annotations/{i_2}.xml"
        #../VOCdevkit/{dsVOC}{year}/Annotations/{i_2}.xml"
        #/data3/chang/ypark/Centernet/centernet-tf2-main-20220719T121658Z-001/centernet-tf2-main/..centernet-tf2-main/VOCdevkit/ceph_VOC2007/Annotations
        #/data3/chang/ypark/Centernet/centernet-tf2-main/VOCdevkit/ceph_VOC2007
        with open(fileName, "wb") as file:
            #try:
            tree.write(file, encoding='utf-8', xml_declaration=True)
            #except:
             #   print(fileName)
              #  pass
            

    print('[Log] xml file has been created.')