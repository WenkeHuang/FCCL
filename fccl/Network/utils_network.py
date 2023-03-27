from fccl.Network.resnet import  ResNet10,ResNet12
from fccl.Network.efficientnet import  EfficientNetB0
from fccl.Network.mobilnet_v2 import MobileNetV2
from fccl.Network.shufflenet import ShuffleNetG2
def init_nets(n_parties,nets_name_list):
    nets_list = {net_i: None for net_i in range(n_parties)}
    for net_i in range(n_parties):
        net_name = nets_name_list[net_i]
        if net_name=='ResNet10':
            net = ResNet10()
        elif net_name =='ResNet12':
            net = ResNet12()
        elif net_name =='Mobilenetv2':
            net = MobileNetV2()
        elif net_name =='Efficientnet':
            net = EfficientNetB0()
        elif net_name =='Shuffnet':
            net = ShuffleNetG2()
        nets_list[net_i] = net
    return nets_list
