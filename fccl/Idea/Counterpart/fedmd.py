import sys
sys.path.append('../../../')
from fccl.Network.utils_network import init_nets
from fccl.Dataset.utils_dataset import init_logs, get_dataloader,generate_public_data_idxs
from fccl.Idea.utils_idea import update_model_via_private_data,evaluate_network,mkdirs
from fccl.Idea.params import args_parser
from sklearn.metrics.pairwise import cosine_similarity
import torch.optim as optim
import torch.nn as nn
from numpy import *
import numpy as np
import torch
import os

args = args_parser()
'''
Global Parameters
'''
Method_Name = 'Counterpart'
Ablation_Name='FedMD'

Temperature = 1
Scenario = args.Scenario
Seed = args.Seed
N_Participants = args.N_Participants
CommunicationEpoch = args.CommunicationEpoch
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize
Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Private_Net_Name_List = args.Private_Net_Name_List
Pariticpant_Params = {
    'loss_funnction' : 'CrossEntropy',
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

def calculate_class_similarity(input_logits,target_logits,sample_size):
    cos_similarity = cosine_similarity(input_logits.cpu(), target_logits.cpu())
    sample_similarity_list = []
    for i in range(sample_size):
        sample_similarity_list.append(cos_similarity[i][i])
    sample_similarity = np.mean(sample_similarity_list)
    return sample_similarity

if __name__ =='__main__':
    logger = init_logs(sub_name=Ablation_Name)
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = args.device_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        priatae_model_path = Project_Dir+'Network/Model_Storage/'+netname+'_'+str(i)+'_'+private_dataset_name+'.ckpt'
        network.load_state_dict(torch.load(priatae_model_path))

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_idxs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length)
    public_train_dl, _, public_train_ds, _ = get_dataloader(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,
    train_bs=TrainBatchSize,test_bs=TestBatchSize,dataidxs=public_data_indexs)

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

            test_dl  = private_test_data_loader_list[participant_index]

            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
        acc_list.append(acc_epoch_list)

        for _ in range(Public_Training_Epoch):

            for batch_idx, (images, _) in enumerate(public_train_dl):
                linear_output_list = []
                linear_output_target_list = []# Save other participants' linear output
                mse_loss_batch_list = []
                '''
                Calculate Linear Output
                '''
                for participant_index in range(N_Participants):
                    network = net_list[participant_index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network.train()
                    images = images.to(device)
                    linear_output = network(x=images)
                    linear_output_target_list.append(linear_output.clone().detach().cpu().numpy().tolist())
                    linear_output_list.append(linear_output)
                linear_output_target_mean = np.mean(linear_output_target_list,axis=0)
                linear_output_target_mean = torch.tensor(linear_output_target_mean).float().to(device)
                '''
                Update Participants' Models via MAE Loss
                '''
                for participant_index in range(N_Participants):
                    network = net_list[participant_index]
                    network = nn.DataParallel(network, device_ids=device_ids).to(device)
                    network.train()
                    criterion = nn.L1Loss(reduction='mean')
                    criterion.to(device)
                    optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                    optimizer.zero_grad()
                    loss =criterion(linear_output_list[participant_index], linear_output_target_mean)
                    mse_loss_batch_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                col_loss_list.append(mse_loss_batch_list)


        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()

            train_dl_local = private_train_data_loader_list[participant_index]

            private_epoch = Private_Training_Epoch[participant_index]
            network, private_loss_batch_list = update_model_via_private_data(network=network,private_epoch=private_epoch,
            private_dataloader=train_dl_local,loss_function=Pariticpant_Params['loss_funnction'],
            optimizer_method=Pariticpant_Params['optimizer_name'],learing_rate=Pariticpant_Params['learning_rate'],
            logger=logger)
            mean_privat_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_privat_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        if epoch_index ==CommunicationEpoch-1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                private_dataset_dir = Dataset_Dir + private_dataset_name
                print(netname+'_'+private_dataset_name+'_'+private_dataset_dir)

                test_dl = private_test_data_loader_list[participant_index]

                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                acc_epoch_list.append(evaluate_network(network=network, dataloader=test_dl, logger=logger))
            acc_list.append(acc_epoch_list)

        if epoch_index % 5 == 0 or epoch_index==CommunicationEpoch-1:
            mkdirs('./Performance_Analysis/'+Scenario)
            mkdirs('./Model_Storage/'+Scenario)
            mkdirs('./Performance_Analysis/'+Scenario+'/'+Ablation_Name)
            mkdirs('./Model_Storage/' +Scenario+'/'+ Ablation_Name)
            mkdirs('./Performance_Analysis/'+Scenario+'/'+Ablation_Name+'/'+Public_Dataset_Name)
            mkdirs('./Model_Storage/' +Scenario+'/'+ Ablation_Name+'/'+Public_Dataset_Name)

            logger.info('Save Loss')
            col_loss_array = np.array(col_loss_list)
            np.save('./Performance_Analysis/' + Scenario +'/' +Ablation_Name+'/'+Public_Dataset_Name
                    +'/collaborative_loss.npy', col_loss_array)
            local_loss_array = np.array(local_loss_list)
            np.save('./Performance_Analysis/'+Scenario+'/'+Ablation_Name+'/'+Public_Dataset_Name
                    +'/local_loss.npy', local_loss_array)
            logger.info('Save Acc')
            acc_array = np.array(acc_list)
            np.save('./Performance_Analysis/' + Scenario +'/' +Ablation_Name+'/'+Public_Dataset_Name
                    +'/acc.npy', acc_array)

            logger.info('Save Models')
            for participant_index in range(N_Participants):
                netname = Private_Net_Name_List[participant_index]
                private_dataset_name = Private_Dataset_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(),
                           './Model_Storage/'+Scenario+'/'+Ablation_Name+'/'+Public_Dataset_Name
                           +'/'+ netname + '_' + str(participant_index) + '_' + private_dataset_name + '.ckpt')
