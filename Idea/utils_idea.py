import copy
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import os
import sys

from tqdm import tqdm
import torchvision.transforms as transforms
sys.path.append('../../')
from Idea.params import args_parser
from PIL import Image
from Idea.third_party import aug
args = args_parser()
Scenario = args.Scenario
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
device_ids = args.device_ids
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Pariticpant_Params = {
    'loss_funnction' : 'KLDivLoss',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
batch_size = 128

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits


def adapt_single(image,net):
    net.eval()
    optimizer = optim.SGD(net.parameters(), lr=Pariticpant_Params['learning_rate'])
    for iteration in range(1):
        inputs = [aug(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss, logits = marginal_entropy(outputs)
        loss.backward()
        optimizer.step()

def memo_evaluate_network(network,teset,logger):
    total = 0
    top1 = 0
    top5 = 0
    
    for i in tqdm(range(len(teset))):
        _, label = teset[i]
        # image = Image.fromarray(teset.data[i])
        image = teset.data[i]
        origin_image = copy.deepcopy(image)
        # label = label.to(device)
        # image = image.to(device)
        adapt_single(image,network)
        network.eval()
        with torch.no_grad():
            outputs = network(origin_image)

            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)
    top1acc= round(100 * top1 / total,2)
    top5acc= round(100 * top5 / total,2)
    logger.info('Accuracy of the network on total {} test images: @top1={}%; @top5={}%'.
            format(total,top1acc,top5acc))
    if Scenario =='Digits':
        return top1acc
    else:
        return top5acc
       

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        total = 0
        top1 = 0
        top5 = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)

            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)

        top1acc= round(100 * top1 / total,2)
        top5acc= round(100 * top5 / total,2)
        logger.info('Accuracy of the network on total {} test images: @top1={}%; @top5={}%'.
              format(total,top1acc,top5acc))
    if Scenario =='Digits':
        return top1acc
    else:
        return top5acc

def evaluate_network_generalization(network,dataloader_list,particiapnt_index,logger):
    generalization_list = []
    network.eval()
    with torch.no_grad():
        for index,dataloader in enumerate(dataloader_list):
            if index != particiapnt_index:
                total = 0
                top1 = 0
                top5 = 0
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = network(images)

                    _, max5 = torch.topk(outputs, 5, dim=-1)
                    labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

                    top1 += (labels == max5[:, 0:1]).sum().item()
                    top5 += (labels == max5).sum().item()
                    total += labels.size(0)

                top1acc = round(100 * top1 / total, 3)
                top5acc = round(100 * top5 / total, 3)
                if Scenario =='Digits':
                    generalization_list.append(top1acc)
                else:
                    generalization_list.append(top5acc)
            else:
                generalization_list.append(0)
    return generalization_list


def update_model_via_private_data(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    criterion = nn.CrossEntropyLoss()
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list

def prox(network,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    criterion = nn.CrossEntropyLoss()
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss.item()))
    return network,participant_local_loss_batch_list


def update_model_via_private_data_with_support_model(network,frozen_network,temperature,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    if loss_function =='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='KLDivLoss':
        criterion = nn.KLDivLoss(reduction='batchmean')
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            with torch.no_grad():
                soft_labels = F.softmax(frozen_network(images)/temperature,dim=1)
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            logsoft_outputs = F.log_softmax(outputs/temperature,dim=1)
            loss_soft = criterion(logsoft_outputs, soft_labels)
            loss_hard = criterion_hard(outputs, labels)
            loss = loss_hard+loss_soft
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_soft: {:.6f} Loss_hard: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), loss_soft.item(),loss_hard.item()))
    return network,participant_local_loss_batch_list

def update_model_via_private_data_with_two_model(network,frozen_network,progressive_network,temperature,private_epoch,private_dataloader,loss_function,optimizer_method,learing_rate,logger):
    if loss_function =='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    if loss_function =='KLDivLoss':
        criterion = nn.KLDivLoss(reduction='batchmean')
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learing_rate)
    criterion_hard = nn.CrossEntropyLoss()
    participant_local_loss_batch_list = []
    for epoch_index in range(private_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            logsoft_outputs = F.log_softmax(outputs/temperature,dim=1)
            with torch.no_grad():
                frozen_soft_labels = F.softmax(frozen_network(images)/temperature,dim=1)
            frozen_loss_soft = criterion(logsoft_outputs, frozen_soft_labels)

            with torch.no_grad():
                progressive_soft_labels = F.softmax(progressive_network(images)/temperature,dim=1)
            progressive_loss_soft = criterion(logsoft_outputs, progressive_soft_labels)

            loss_hard = criterion_hard(outputs, labels)

            loss = loss_hard+frozen_loss_soft+progressive_loss_soft
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch_index % 5 ==0:
                logger.info('Private Train Epoch: {} [{}/{} ({:.0f}%)]\tfrozen_loss_soft: {:.6f} progressive_loss_soft: {:.6f} Loss_hard: {:.6f}'.format(
                    epoch_index, batch_idx * len(images), len(private_dataloader.dataset),
                           100. * batch_idx / len(private_dataloader), frozen_loss_soft.item(),progressive_loss_soft.item(),loss_hard.item()))
    return network,participant_local_loss_batch_list
