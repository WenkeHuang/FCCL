import sys
sys.path.append("..")
sys.path.append('.')
from Dataset.utils_dataset import init_logs, get_dataloader
from Network.utils_network import init_nets
from Idea.params import args_parser
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import manifold
import torch.nn as nn
import numpy as np
from numpy import *
import torch
import os

args = args_parser()
Seed = args.Seed
N_Participants = args.N_Participants
TrainBatchSize = args.TrainBatchSize
Local_TrainBatchSize = args.Local_TrainBatchSize
TestBatchSize = args.TestBatchSize
Pretrain_Epoch = 2
Federated_learning_Option = True
Original_Path = args.Original_Path
Pariticpant_Params = {
    'loss_funnction' : 'CrossEntropy',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}

Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Private_Net_Name_List = args.Private_Net_Name_List
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Data_Total_Len_List = args.Private_Data_Total_Len_List
Private_Data_Len_List = args.Private_Data_Len_List
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)

def pretrain_network(epoch,net,data_loader,loss_function,optimizer_name,learning_rate):
    if loss_function =='CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if optimizer_name =='Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    for _epoch in range(epoch):
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                _epoch, batch_idx * len(images), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader), loss.item()))
    return net

def evaluate_network(net,dataloader):
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        top1 = 0
        top5 = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            _, max5 = torch.topk(outputs, 5, dim=-1)
            labels = labels.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]

            top1 += (labels == max5[:, 0:1]).sum().item()
            top5 += (labels == max5).sum().item()
            total += labels.size(0)

        top1acc= round(100 * top1 / total,2)
        top5acc= round(100 * top5 / total,2)
        print('Accuracy of the network on total {} test images: @top1={}%; @top5={}%'.
              format(total,top1acc,top5acc))
    return 100 * correct / total

def tsne_network(net,dataloader,model_name,classes):
    with torch.no_grad():
        logit_list = []
        label_list = []
        for images, labels in dataloader:
            images = images.to(device)
            outputs = net(images)
            outputs = outputs.cpu().numpy().tolist()
            labels = labels.numpy().tolist()
            logit_list.extend(outputs)
            label_list.extend(labels)
        logit_array = np.array(logit_list).flatten().reshape(-1, Output_Channel)
        label_array = np.array(label_list).flatten()
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        X_tsne = tsne.fit_transform(logit_array)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        scatter  = plt.scatter(X_norm[:,0], X_norm[:, 1], cmap=plt.cm.tab10,c=label_array)
        plt.legend(handles=scatter.legend_elements()[0], labels=classes)
        plt.xticks([])
        plt.yticks([])
        plt.savefig( Original_Path+'Model_Visualization/tsne_'+model_name+'.png',bbox_inches='tight')

if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3,4"
    device_ids = args.device_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info("Load Participants' Data and Model")
    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    private_dataset_idxs_dict = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(Private_Data_Total_Len_List[index])
        idxes = idxes[0:Private_Data_Len_List[index]]
        private_dataset_idxs_dict[Private_Dataset_Name_List[index]]= idxes
    logger.info(private_dataset_idxs_dict)

    if Federated_learning_Option:
        logger.info('Pretrain Participants Models')
        for index in range(N_Participants):
            
            print('The index is'+str(index))
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_idx = private_dataset_idxs_dict[participant_dataset_name]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name

            train_dl_local ,test_dl, train_ds_local, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=Local_TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=participant_idx)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            logger.info('Pretrain the '+str(index)+' th Participant Model with N_training: '+str(len(train_ds_local)))
            # logger.info(
            #     'Load the '+str(index)+' th Participant Model')
            # network.load_state_dict(torch.load('./Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
            network = pretrain_network(epoch=Pretrain_Epoch,net=network,data_loader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
                                       optimizer_name=Pariticpant_Params['optimizer_name'],learning_rate=Pariticpant_Params['learning_rate'])
            evaluate_network(net=network,dataloader=test_dl)
            logger.info('Save the '+str(index)+' th Participant Model')
            torch.save(network.state_dict(), Original_Path+'/Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt')

        logger.info('Evaluate Models')
        test_accuracy_list = []
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _ ,test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            network.load_state_dict(torch.load(Original_Path+'/Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
            output = evaluate_network(net=network,dataloader=test_dl)
            test_accuracy_list.append(output)
        print('The average Accuracy of models on the test images:'+str(mean(test_accuracy_list)))

        logger.info('Visualization Models Performance')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _ ,test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            network.load_state_dict(torch.load(Original_Path+'/Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
            tsne_network(net=network, dataloader=test_dl, model_name=netname+'_'+str(index)+'_'+participant_dataset_name, classes=Private_Dataset_Classes)
    else:
        logger.info('Pretrain Global Models')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            train_dl_local ,test_dl, train_ds_local, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            logger.info(
                'Pretrain the '+str(index)+' th Global Model with N_training: ' + str(len(train_ds_local)))
            # logger.info(
            #     'Load the '+str(index)+' th Global Model')
            network = pretrain_network(epoch=Pretrain_Epoch, net=network, data_loader=train_dl_local,
                                       loss_function=Pariticpant_Params['loss_funnction'],
                                       optimizer_name=Pariticpant_Params['optimizer_name'],
                                       learning_rate=Pariticpant_Params['learning_rate'])
            logger.info('Save the Global Model')
            torch.save(network.state_dict(), Original_Path+'/Model_Storage/' + netname + '_Global_'+participant_dataset_name+ '.ckpt')

        logger.info('Evaluate Global Models')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _, test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            network.load_state_dict(torch.load(Original_Path+'/Model_Storage/' + netname + '_Global_'+participant_dataset_name+ '.ckpt'))
            evaluate_network(net=network, dataloader=test_dl)

        logger.info('Visualization Global Models Performance')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _, test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Private_Net_Name_List[index]
            tsne_network(net=network, dataloader=test_dl, model_name=netname + '_Global_'+participant_dataset_name, classes=Private_Dataset_Classes)