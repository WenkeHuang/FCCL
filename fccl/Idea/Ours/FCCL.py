import sys

sys.path.append('../../../')
sys.path.append('')
from fccl.Network.utils_network import init_nets
from fccl.Dataset.utils_dataset import init_logs, get_dataloader,generate_public_data_idxs
from fccl.Idea.utils_idea import update_model_via_private_data_with_two_model,evaluate_network,mkdirs
from fccl.Idea.params import args_parser
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
from numpy import *
import numpy as np
import torch
import copy
import os

args = args_parser()

'''
Global Parameters
'''
Method_Name = 'Ours'
Ablation_Name='FCCL'

Temperature = 1
Scenario = args.Scenario
Seed = args.Seed
N_Participants = args.N_Participants
CommunicationEpoch = args.CommunicationEpoch
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize
Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Idea_Ours_Dir = args.Project_Dir + 'Idea/Ours/'
Private_Net_Name_List = args.Private_Net_Name_List
Pariticpant_Params = {
    'loss_funnction' : 'KLDivLoss',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
'''
Scenario for large domain gap
'''
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Data_Total_Len_List = args.Private_Data_Total_Len_List
Private_Data_Len_List = args.Private_Data_Len_List
Private_Training_Epoch = args.Private_Training_Epoch
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)
'''
Public data parameters
'''
Public_Dataset_Name = args.Public_Dataset_Name
Public_Dataset_Length = args.Public_Dataset_Length
Public_Dataset_Dir = Dataset_Dir+Public_Dataset_Name
Public_Training_Epoch = args.Public_Training_Epoch


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ =='__main__':
    logger = init_logs(sub_name=Ablation_Name)
    logger.info('Method Name : '+Method_Name + ' Ablation Name : '+Ablation_Name)
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
    device_ids = args.device_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info("Initialize Participants' Data idxs and Model")
    # For Digits scenario
    private_dataset_idxs_dict = {}
    for index in range(N_Participants):
        idxes = np.random.permutation(Private_Data_Total_Len_List[index])
        idxes = idxes[0:Private_Data_Len_List[index]]
        private_dataset_idxs_dict[Private_Dataset_Name_List[index]]= idxes
    logger.info(private_dataset_idxs_dict)


    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Participants' Models")
    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    frozen_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Frozen Participants' Models")
    for i in range(N_Participants):
        network = frozen_net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    progressive_net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    logger.info("Load Progressive Participants' Models")
    for i in range(N_Participants):
        network = progressive_net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[i]
        private_dataset_name = Private_Dataset_Name_List[i]
        private_model_path = Project_Dir + 'Network/Model_Storage/' + netname + '_' + str(i) + '_' + private_dataset_name + '.ckpt'
        network.load_state_dict(torch.load(private_model_path))

    logger.info("Initialize Public Data Parameters")
    print(Scenario+Public_Dataset_Name)
    public_data_indexs = generate_public_data_idxs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length)

    public_train_dl, _, _, _ = get_dataloader(dataset=Public_Dataset_Name, datadir=Public_Dataset_Dir,
                                                            train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                            dataidxs=public_data_indexs)
    logger.info('Initialize Private Data Loader')
    private_train_data_loader_list = []
    private_test_data_loader_list = []
    for participant_index in range(N_Participants):
        private_dataset_name = Private_Dataset_Name_List[participant_index]
        private_dataidx = private_dataset_idxs_dict[private_dataset_name]
        private_dataset_dir = Dataset_Dir + private_dataset_name
        train_dl_local, test_dl_local, _, _ = get_dataloader(dataset=private_dataset_name,
                                                 datadir=private_dataset_dir,
                                                 train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                 dataidxs=private_dataidx)
        private_train_data_loader_list.append(train_dl_local)
        private_test_data_loader_list.append(test_dl_local)

    col_loss_list = []
    local_loss_list = []
    acc_list = []
    for epoch_index in range(CommunicationEpoch):

        logger.info("The "+str(epoch_index)+" th Communication Epoch")
        logger.info('Evaluate Models')
        acc_epoch_list = []
        for participant_index in range(N_Participants):
            netname = Private_Net_Name_List[participant_index]
            private_dataset_name = Private_Dataset_Name_List[participant_index]
            private_dataset_dir = Dataset_Dir + private_dataset_name
            print(netname + '_' + private_dataset_name + '_' + private_dataset_dir)
            _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                              train_bs=TrainBatchSize,
                                              test_bs=TestBatchSize, dataidxs=None)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
        acc_list.append(acc_epoch_list)

        a = datetime.now()
        for _ in range(Public_Training_Epoch):
            for batch_idx, (images, _) in enumerate(public_train_dl):
                linear_output_list = []
                linear_output_target_list = [] # Save other participants' linear output
                linear_output_progressive_list = [] # Save itself progressive model's linear output
                col_loss_batch_list = []
                '''
                Calculate Linear Output
                '''
                for participant_index in range(N_Participants):
                    network = net_list[participant_index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network.train()
                    images = images.to(device)
                    linear_output = network(x=images)
                    linear_output_target_list.append(linear_output.clone().detach())
                    linear_output_list.append(linear_output)
                '''
                Calculate Progressive Linear Output
                '''
                for progressive_participant_index in range(N_Participants):
                    progressive_network = progressive_net_list[progressive_participant_index]
                    progressive_network = nn.DataParallel(progressive_network,device_ids=device_ids).to(device)
                    progressive_network.eval()
                    with torch.no_grad():
                        images = images.to(device)
                        progressive_linear_output = progressive_network(images)
                        linear_output_progressive_list.append(progressive_linear_output)
                '''
                Update Participants' Models via Col Loss
                '''
                for participant_index in range(N_Participants):
                    '''
                    Calculate the Loss with others
                    '''
                    network = net_list[participant_index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network.train()
                    optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                    optimizer.zero_grad()
                    linear_output_target_avg_list = []
                    for i in range(N_Participants):
                        if i != participant_index:
                            linear_output_target_avg_list.append(linear_output_target_list[i])
                        if i ==participant_index:
                            linear_output_target_avg_list.append(linear_output_progressive_list[i])

                    linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)
                    linear_output = linear_output_list[participant_index]
                    z_1_bn = (linear_output-linear_output.mean(0))/linear_output.std(0)
                    z_2_bn = (linear_output_target_avg-linear_output_target_avg.mean(0))/linear_output_target_avg.std(0)
                    # empirical cross-correlation matrix
                    c = z_1_bn.T @ z_2_bn
                    # sum the cross-correlation matrix between all gpus
                    c.div_(len(images))

                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = off_diagonal(c).add_(1).pow_(2).sum()
                    col_loss = on_diag + 0.0051 * off_diag
                    col_loss_batch_list.append(col_loss.item())
                    col_loss.backward()
                    optimizer.step()
                col_loss_list.append(col_loss_batch_list)

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()

            frozen_network = frozen_net_list[participant_index]
            frozen_network = nn.DataParallel(frozen_network,device_ids=device_ids).to(device)
            frozen_network.eval()

            progressive_network = progressive_net_list[participant_index]
            progressive_network = nn.DataParallel(progressive_network,device_ids=device_ids).to(device)
            progressive_network.eval()

            private_dataset_name=  Private_Dataset_Name_List[participant_index]
            private_dataidx = private_dataset_idxs_dict[private_dataset_name]
            private_dataset_dir = Dataset_Dir+private_dataset_name
            train_dl_local, _, train_ds_local, _ = get_dataloader(dataset=private_dataset_name,
                                                                  datadir=private_dataset_dir,
                                                                  train_bs=TrainBatchSize, test_bs=TestBatchSize,
                                                                  dataidxs=private_dataidx)

            private_epoch = max(int(Public_Dataset_Length / len(train_ds_local)), 1)
            private_epoch = Private_Training_Epoch[participant_index]

            network, private_loss_batch_list = update_model_via_private_data_with_two_model(network=network,
            frozen_network=frozen_network,progressive_network=progressive_network,
            temperature = Temperature,private_epoch=private_epoch,private_dataloader=train_dl_local,
            loss_function=Pariticpant_Params['loss_funnction'],optimizer_method=Pariticpant_Params['optimizer_name'],
            learing_rate=Pariticpant_Params['learning_rate'],logger=logger)
            mean_private_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        b = datetime.now()
        temp = b-a
        print(temp)
        '''
        用于迭代 Progressive 模型
        '''
        for j in range(N_Participants):
            progressive_net_list[j] = copy.deepcopy(net_list[j])

        if epoch_index ==CommunicationEpoch-1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                private_dataset_dir = Dataset_Dir + private_dataset_name
                print(netname+'_'+private_dataset_name+'_'+private_dataset_dir)
                _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                                  train_bs=TrainBatchSize,
                                                  test_bs=TestBatchSize, dataidxs=None)
                
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                
                acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
            acc_list.append(acc_epoch_list)

        if epoch_index % 5 == 3 or epoch_index == CommunicationEpoch - 1:
            mkdirs(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario)
            mkdirs(Idea_Ours_Dir + '/Model_Storage/' + Scenario)
            mkdirs(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario + '/' + Ablation_Name)
            mkdirs(Idea_Ours_Dir + '/Model_Storage/' + Scenario + '/' + Ablation_Name)
            mkdirs(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)
            mkdirs(Idea_Ours_Dir + '/Model_Storage/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name)

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save(Idea_Ours_Dir + '/Performance_Analysis/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                    + '/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           Idea_Ours_Dir + '/Model_Storage/' + Scenario + '/' + Ablation_Name + '/' + Public_Dataset_Name
                           + '/' + netname + '_' + str(participant_index) + '_' + private_dataset_name + '.ckpt')
