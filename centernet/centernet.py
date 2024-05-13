from ast import Param
import colorsys
import os
import time

import numpy as np
import pandas as pd # celes
import tensorflow as tf
from PIL import ImageDraw, ImageFont

from nets.centernet import centernet
from utils.utils import (cvtColor, get_classes, preprocess_input, resize_image,
                         show_config)
from utils.utils_bbox import BBoxUtility


#--------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、classes_path和backbone
#   都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class CenterNet(object):
    _defaults = {
        # "model_path"        : 'model_data/centernet_resnet50_ceph_voc.h5',
        #"model_path"        : 'logs/best_epoch_weights.h5', # dsaint31
        # "model_path"        : '/data3/chang/ypark/Centernet/centernet-tf2-main-20220719T121658Z-001/centernet-tf2-main/logs/best_epoch_weights.h5',
        
        "model_path"        : './logs/best_epoch_weights.h5',
        "classes_path"      : '/data3/chang/ypark/Centernet/centernet-tf2-main-20220719T121658Z-001/centernet-tf2-main/model_data/voc_classes.txt',
        #/data3/chang/ypark/Centernet/centernet-tf2-main-20220719T121658Z-001/centernet-tf2-main/model_data/voc_classes.txt
        "input_shape"       : [256,256],
        "backbone"          : 'resnet50',
        "confidence"        : 0.3,
        "nms_iou"           : 0.3,
        "nms"               : True,
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化centernet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.heatmap = False
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #   计算总的类的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.bbox_util = BBoxUtility(nms_thresh=self.nms_iou)
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   载入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        #----------------------------------------#
        #   创建centernet模型
        #----------------------------------------#
        self.centernet = centernet([self.input_shape[0], self.input_shape[1], 3], num_classes=self.num_classes, backbone=self.backbone, mode='predict' if not self.heatmap else 'heatmap')
        self.centernet.load_weights(self.model_path, by_name=True)

    @tf.function
    def get_pred(self, photo):
        preds = self.centernet(photo, training=False)
        return preds
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        if results[0] is None:
            return image

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)

        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        
        check_top_score = {} #dsaint31
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            # dsaint31 -----------------------------
            fscore = float(score)
            if fscore <= check_top_score.get(predicted_class,0.):
                continue
            check_top_score[predicted_class] = fscore
            # end ---------------------------------


            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            # print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image





    #### celes ##########################################################################################3###


    def detect_image_ceph(self, image, image_save = False, text=False):

        ret_dict = {}
        ret_img = None
        point_array= []
        
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        if results[0] is None:
            return None

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)


        check_top_score = {} #dsaint31
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            # dsaint31 -----------------------------
            fscore = float(score)
            if fscore <= check_top_score.get(predicted_class,0.):
                continue
            check_top_score[predicted_class] = fscore
            # end ---------------------------------


            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            ##################
            # original 크기 기준으로 좌표 저장
            #point_array.append([predicted_class, ((left+right)/2) * (image.size[1]/self.input_shape[1]), ((top+bottom)/2) * (image.size[0]/self.input_shape[0])]) # celes
            try:
                tmp_x = ((left+right)/2)*(image.size[1]/self.input_shape[1])
                tmp_y = ((top+bottom)/2)*(image.size[0]/self.input_shape[0])
                ret_dict[predicted_class] = (tmp_x,tmp_y)
                # point_dict = {predicted_class, 'coordinates':(,)}

            except:
                #ret_dict[predicted_class] = (np.nan,np.nan)
                ret_dict[predicted_class] = (-1, -1) # celes
                # point_dict = {predicted_class : 'x_coordinate':'-1', 'y_coordinate':'-1'}

            if image_save==True:

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                #print(label, top, left, bottom, right)
            
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    param = 3
                    draw.ellipse([((left+right)/2, (top+bottom)/2), (param+(left+right)/2, param+(top+bottom)/2)], outline=self.colors[c], width=2)

                if text==True:
                    font_2 = ImageFont.truetype(font='model_data/simhei.ttf', size=10)
                    draw.text(text_origin, str(label,'UTF-8'), fill=self.colors[c], font=font_2, anchor='md')
                del draw

        if image_save==True:
            ret_img = image


            
        point_array.append(ret_dict)
        point_array.append(ret_img)
        #print(ret_dict, ret_img)
        return point_array

        
        

    ##########################################################################################################################





    def get_FPS(self, image, test_interval):
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        #--------------------------------------------------------------------------------------------#
        #   centernet后处理的过程，包括门限判断和传统非极大抑制。
        #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
        #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
        #   这里面存在传统的nms处理方法，可以选择关闭和开启。
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        #--------------------------------------------------------------------------------------------#
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t1 = time.time()
        for _ in range(test_interval):
            outputs = self.get_pred(image_data).numpy()
            #--------------------------------------------------------------------------------------------#
            #   centernet后处理的过程，包括门限判断和传统非极大抑制。
            #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
            #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
            #   这里面存在传统的nms处理方法，可以选择关闭和开启。
            #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
            #--------------------------------------------------------------------------------------------#
            results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt

        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，并进行归一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        output  = self.centernet.predict(image_data)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask        = np.zeros((image.size[1], image.size[0]))
        score       = np.max(output[0], -1)
        score       = cv2.resize(score, (image.size[0], image.size[1]))
        normed_score    = (score * 255).astype('uint8')
        mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()
        
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------#
        #   获得输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度，图片预处理，归一化。
        #---------------------------------------------------------#
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        outputs    = self.get_pred(image_data).numpy()
        #--------------------------------------------------------------------------------------------#
        #   centernet后处理的过程，包括门限判断和传统非极大抑制。
        #   对于centernet网络来讲，确立中心非常重要。对于大目标而言，会存在许多的局部信息。
        #   此时大目标中心点比较难以确定。使用最大池化的非极大抑制方法无法去除局部框
        #   这里面存在传统的nms处理方法，可以选择关闭和开启。
        #   实际测试中，hourglass为主干网络时有无额外的nms相差不大，resnet相差较大。
        #--------------------------------------------------------------------------------------------#
        results = self.bbox_util.postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, confidence=self.confidence)

        #--------------------------------------#
        #   如果没有检测到物体，则返回原图
        #--------------------------------------#
        if results[0] is None:
            return 

        top_label   = np.array(results[0][:, 5], dtype = 'int32')
        top_conf    = results[0][:, 4]
        top_boxes   = results[0][:, :4]

        check_top_score = {} #dsaint31
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])
            
            # dsaint31 -----------------
            fscore = float(score)
            if fscore <= check_top_score.get(predicted_class,0.):
                continue
            check_top_score[predicted_class] = fscore
            # end -----------------------
            
            top, left, bottom, right = box

            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 