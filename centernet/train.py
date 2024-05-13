import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.centernet import centernet
from nets.centernet_training import get_lr_scheduler
from utils.callbacks import EvalCallback, LossHistory, ModelCheckpoint
from utils.dataloader import CenternetDatasets
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    eager           = False
    train_gpu       = [1,2]
    classes_path    = './model_data/voc_classes.txt' # dsaint31
    # model_path      = 'model_data/centernet_resnet50_voc.h5'
    model_path      = './model_data/centernet_resnet50_ceph_voc.h5' #dsaint31
    model_path      = './logs/best_epoch_weights.h5' #dsaint31
    model_path = ''
    input_shape     = [256,256] #1000,1000
    backbone        = "resnet50"
    Init_Epoch          = 0
    Freeze_Epoch        = 100
    Freeze_batch_size   = 16 #16 
    UnFreeze_Epoch      = 150
    Unfreeze_batch_size = 8#8 
    Freeze_Train        = True
    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 5
    num_workers         = 1
    train_annotation_path   = '/data3/chang/ypark/Centernet/2007_train.txt'
    val_annotation_path     = '/data3/chang/ypark/Centernet/2007_val.txt'

    os.environ["CUDA_VISIBLE_DEVICES"]  = ','.join(str(x) for x in train_gpu)
    ngpus_per_node                      = len(train_gpu)
    
    #gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    gpus = tf.config.list_physical_devices(device_type='GPU') # dsaint31
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    #------------------------------------------------------#
    #   判断当前使用的GPU数量与机器上实际的GPU数量
    #------------------------------------------------------#
    if ngpus_per_node > 1 and ngpus_per_node > len(gpus):
        raise ValueError("The number of GPUs specified for training is more than the GPUs on the machine")
        
    if ngpus_per_node > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None
    print('Number of devices: {}'.format(ngpus_per_node))

    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    #----------------------------------------------------#
    #   判断是否多GPU载入模型和预训练权重
    #----------------------------------------------------#
    if ngpus_per_node > 1:
        with strategy.scope():
            model, prediction_model = centernet([input_shape[0], input_shape[1], 3], num_classes=num_classes, backbone=backbone, mode='train')
            if model_path != '':
                #------------------------------------------------------#
                #   载入预训练权重
                #------------------------------------------------------#
                print('Load weights {}.'.format(model_path))
                model.load_weights(model_path, by_name=True, skip_mismatch=True)
    else:
        model, prediction_model = centernet([input_shape[0], input_shape[1], 3], num_classes=num_classes, backbone=backbone, mode='train')
        if model_path != '':
            #------------------------------------------------------#
            #   载入预训练权重
            #------------------------------------------------------#
            print('Load weights {}.'.format(model_path))
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
        
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
        classes_path = classes_path, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )
    #---------------------------------------------------------#
    #   总训练世代指的是遍历全部数据的总次数
    #   总训练步长指的是梯度下降的总次数 
    #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
    #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
    #----------------------------------------------------------#
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step //(num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            if backbone == "resnet50":
                freeze_layer = 150
            elif backbone == "hourglass":
                freeze_layer = 600
            else:
                raise ValueError('Unsupported backbone - `{}`, Use resnet50, hourglass.'.format(backbone))

            for i in range(freeze_layer): model.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layer, len(model.layers)))
            
        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
        lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('The dataset is too small for training, please expand the dataset.')

        train_dataloader    = CenternetDatasets(train_lines, input_shape, batch_size, num_classes, train = True)
        val_dataloader      = CenternetDatasets(val_lines, input_shape, batch_size, num_classes, train = False)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False

            gen         = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val     = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
            if ngpus_per_node > 1:
                gen     = strategy.experimental_distribute_dataset(gen)
                gen_val = strategy.experimental_distribute_dataset(gen_val)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            eval_callback   = EvalCallback(prediction_model, input_shape, class_names, num_classes, val_lines, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)
            #---------------------------------------#
            #   开始模型训练
            #---------------------------------------#
            for epoch in range(start_epoch, end_epoch):
                #---------------------------------------#
                #   如果模型有冻结学习部分
                #   则解冻，并设置参数
                #---------------------------------------#
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    #-------------------------------------------------------------------#
                    #   判断当前batch_size，自适应调整学习率
                    #-------------------------------------------------------------------#
                    nbs             = 64
                    lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
                    lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                    Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                    Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                    #---------------------------------------#
                    #   获得学习率下降的公式
                    #---------------------------------------#
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model.layers)): 
                        model.layers[i].trainable = True

                    epoch_step      = num_train // batch_size
                    epoch_step_val  = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                    train_dataloader.batch_size    = batch_size
                    val_dataloader.batch_size      = batch_size

                    gen         = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
                    gen_val     = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))

                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    if ngpus_per_node > 1:
                        gen     = strategy.experimental_distribute_dataset(gen)
                        gen_val = strategy.experimental_distribute_dataset(gen_val)
                    
                    UnFreeze_flag = True
                    
                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)
                
                fit_one_epoch(model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, save_period, save_dir, strategy)
                
                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
            if ngpus_per_node > 1:
                with strategy.scope():
                    model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
            else:
                model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
                
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint(os.path.join(save_dir, "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            checkpoint_last = ModelCheckpoint(os.path.join(save_dir, "last_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
            checkpoint_best = ModelCheckpoint(os.path.join(save_dir, "best_epoch_weights.h5"), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = True, period = 1)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            eval_callback   = EvalCallback(prediction_model, input_shape, class_names, num_classes, val_lines, log_dir, \
                                            eval_flag=eval_flag, period=eval_period)
            callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch
                    
                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 64
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 5e-2
                lr_limit_min    = 2.5e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, checkpoint_last, checkpoint_best, lr_scheduler, eval_callback]
                    
                for i in range(len(model.layers)): 
                    model.layers[i].trainable = True
                if ngpus_per_node > 1:
                    with strategy.scope():
                        model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})
                else:
                    model.compile(optimizer = optimizer, loss = {'centernet_loss': lambda y_true, y_pred: y_pred})

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                train_dataloader.batch_size    = Unfreeze_batch_size
                val_dataloader.batch_size      = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit(
                    x                   = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
