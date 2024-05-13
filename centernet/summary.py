#--------------------------------------------#
#   This code to dispaly the network structure
#--------------------------------------------#
from nets.centernet import centernet
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [512,512] #512,512
    num_classes     = 20
    
    model, _ = centernet([input_shape[0], input_shape[1], 3], num_classes, backbone='resnet50')
    #--------------------------------------------#
    #   View Network Structure Network Structure
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   FLOPS for computing networks
    #--------------------------------------------#
    net_flops(model, table=False)
    
    #--------------------------------------------#
    #   Obtain the name and serial number of each layer of the network
    #--------------------------------------------#
    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
