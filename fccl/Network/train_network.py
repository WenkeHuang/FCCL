import sys
sys.path.append("../..")
from fccl.Dataset.utils_dataset import init_logs, get_dataloader
from fccl.Network.utils_network import init_nets
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import manifold
import torch.nn as nn
from numpy import *
import torch
import os

Seed = 0
N_Participants = 4
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
Federated_learning_Option = True

Pariticpant_Params = {
    'loss_funnction' : 'CrossEntropy',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}

Nets_Name_List = ['ResNet10','ResNet12','Efficientnet','Mobilenetv2']
Private_Dataset_Name_List = ['mnist','usps','svhn','syn']
Private_Data_Total_Len_List = [60000, 7291, 73257, 10000]
Private_Data_Len_List = [150,80,5000,1800]
Dataset_Dir = r'/data0/federated_learning/'
Private_Dataset_Classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
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
            # print(images.size())
            # plt.figure(figsize=(9, 9))
            # for i in range(9):
            #     plt.subplot(3, 3, i + 1)
            #     plt.title(labels[i].item())
            #     plt.imshow(images[i].permute(1, 2, 0))
            #     plt.axis('off')
            # plt.savefig('demo.png')
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
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
        plt.savefig('./Model_Visualization/tsne_'+model_name+'.png')

if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = [0,1,2,3,4,5,6,7]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info("Load Participants' Data and Model")
    net_list = init_nets(n_parties=N_Participants,nets_name_list=Nets_Name_List)
    private_dataset_idxs_dict = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(Private_Data_Total_Len_List[index])
        idxes = idxes[0:Private_Data_Len_List[index]]
        private_dataset_idxs_dict[Private_Dataset_Name_List[index]]= idxes
    logger.info(private_dataset_idxs_dict)

    if Federated_learning_Option:
        logger.info('Pretrain Participants Models')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_idx = private_dataset_idxs_dict[participant_dataset_name]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name

            train_dl_local ,test_dl, train_ds_local, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=participant_idx)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Nets_Name_List[index]
            logger.info('Pretrain the '+str(index)+' th Participant Model with N_training: '+str(len(train_ds_local)))
            # logger.info(
            #     'Load the '+str(index)+' th Participant Model')
            # network.load_state_dict(torch.load('./Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
            network = pretrain_network(epoch=Pretrain_Epoch,net=network,data_loader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
                                       optimizer_name=Pariticpant_Params['optimizer_name'],learning_rate=Pariticpant_Params['learning_rate'])
            evaluate_network(net=network,dataloader=test_dl)
            logger.info('Save the '+str(index)+' th Participant Model')
            torch.save(network.state_dict(), './Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt')

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
            netname = Nets_Name_List[index]
            network.load_state_dict(torch.load('./Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
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
            netname = Nets_Name_List[index]
            network.load_state_dict(torch.load('./Model_Storage/'+netname+'_'+str(index)+'_'+participant_dataset_name+'.ckpt'))
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
            netname = Nets_Name_List[index]
            logger.info(
                'Pretrain the '+str(index)+' th Global Model with N_training: ' + str(len(train_ds_local)))
            # logger.info(
            #     'Load the '+str(index)+' th Global Model')
            # network.load_state_dict(torch.load('./Model_Storage/' + netname + '_Global' + '.ckpt'))
            network = pretrain_network(epoch=Pretrain_Epoch, net=network, data_loader=train_dl_local,
                                       loss_function=Pariticpant_Params['loss_funnction'],
                                       optimizer_name=Pariticpant_Params['optimizer_name'],
                                       learning_rate=Pariticpant_Params['learning_rate'])
            logger.info('Save the Global Model')
            # torch.save(network.state_dict(), './Model_Storage/' + netname + '_Global' + '.ckpt')
            torch.save(network.state_dict(), './Model_Storage/' + netname + '_Global_'+participant_dataset_name+ '.ckpt')

        logger.info('Evaluate Global Models')
        for index in range(N_Participants):
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _, test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Nets_Name_List[index]
            # network.load_state_dict(torch.load('./Model_Storage/' + netname + '_Global' + '.ckpt'))
            network.load_state_dict(torch.load('./Model_Storage/' + netname + '_Global_'+participant_dataset_name+ '.ckpt'))
            evaluate_network(net=network, dataloader=test_dl)

        logger.info('Visualization Global Models Performance')
        for index in range(N_Participants):  # 改成2 拿来测试 N_Participants
            participant_dataset_name = Private_Dataset_Name_List[index]
            participant_dataset_dir = Dataset_Dir+participant_dataset_name
            _, test_dl, _, _ = get_dataloader(
            dataset=participant_dataset_name,datadir=participant_dataset_dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                dataidxs=None)
            network = net_list[index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            netname = Nets_Name_List[index]
            # network.load_state_dict(torch.load('./Model_Storage/' + netname + '_Global' + '.ckpt'))
            tsne_network(net=network, dataloader=test_dl, model_name=netname + '_Global_'+participant_dataset_name, classes=Private_Dataset_Classes)