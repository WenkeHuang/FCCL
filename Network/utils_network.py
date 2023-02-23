
from Network.Resnet import  ResNet10,ResNet12,ResNet18,ResNet34
from Network.Efficientnet import  EfficientNetB0
from Network.Mobilnet_v2 import MobileNetV2

# from Network.resnet import  ResNet10,ResNet12,ResNet18,ResNet34
# from Network.efficientnet import  EfficientNetB0
# from Network.mobilnet_v2 import MobileNetV2
from Network.googlenet import GoogLeNet

def init_nets(n_parties,nets_name_list,num_classes=10):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10(num_classes=num_classes)
        elif net_name =='ResNet12':
            net = ResNet12(num_classes=num_classes)
        elif net_name =='ResNet18':
            net = ResNet18(num_classes=num_classes)
        elif net_name =='ResNet34':
            net = ResNet34(num_classes=num_classes)
        elif net_name =='Mobilenetv2':
            net = MobileNetV2(num_classes=num_classes)
        elif net_name =='Googlenet':
            net = GoogLeNet(num_classes=num_classes)
        elif net_name =='Efficientnet':
            net = EfficientNetB0(num_classes=num_classes)

        nets_list[net_i] = net
    return nets_list