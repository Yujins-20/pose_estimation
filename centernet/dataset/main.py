from txt_generate import textwrite
from xml_generate import xml_writer

import pandas as pd, os

def __init__(self):
    self.label_data = None
    self.dsVOC = 'ceph_VOC' #'VOC'
    self.year = '2007'

def main():
    # parameters for file creation
    dsVOC = 'ceph_VOC' #'VOC'
    year = '2007'

    # csv load
    #csv_name = input('Type your csv name (ex: label_data.csv) : ')

    try:
        label_path = "/data3/chang/ypark/Centernet/data/resize/csv_img/front/f2_all.csv" #####

        if os.path.exists(label_path)==True:
            label_data = pd.read_csv("/data3/chang/ypark/Centernet/data/resize/csv_img/front/f2_all.csv")
          

        elif os.path.exists(label_path)==False:
            label_data = pd.read_csv("/data3/chang/ypark/Centernet/data/resize/csv_img/front/f2_all.csv") #/content/drive/MyDrive/labels.csv
            


        print('\n >>> csv file has been loaded.\n\n ================================')


        os.makedirs(f'../centernet-tf2-main/VOCdevkit/{dsVOC}{year}/JPEGImages/', exist_ok=True)
        os.makedirs(f'../centernet-tf2-main/VOCdevkit/{dsVOC}{year}/Annotations/', exist_ok=True)
        os.makedirs(f'../centernet-tf2-main/VOCdevkit/{dsVOC}{year}/ImageSets/Main/', exist_ok=True)

        print('[Log] Directory has been created.')

        xml_writer(label_data, dsVOC, year)
        textwrite(label_data, dsVOC, year)
        


    except ValueError:
        print('\n >>> csv file does not exist. \n\n ')

    print('================================ \n\n >>> Done')


main()