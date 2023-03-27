import sys
sys.path.append("../..")
sys.path.append("")
from fccl.Network.utils_network import init_nets
from fccl.Dataset.utils_dataset import init_logs, get_dataloader
from fccl.Idea.utils_idea import evaluate_network_generalization
from fccl.Idea.params import args_parser

import torch
import torch.nn as nn
from numpy import *
import os



Evaluation_Idea_Dict = {
    'Ours':['FCCL'],
    'Counterpart': ['FedMD'],
}

args = args_parser()

Scenario = args.Scenario
Idea_Dir = args.Project_Dir+'Idea/'
Original_Path = args.Original_Path
Seed = args.Seed

N_Participants = args.N_Participants
TrainBatchSize = args.TrainBatchSize
TestBatchSize = args.TestBatchSize

Dataset_Dir = args.Dataset_Dir
Project_Dir = args.Project_Dir
Private_Net_Name_List = args.Private_Net_Name_List
Private_Dataset_Classes = args.Private_Dataset_Classes
Output_Channel = len(Private_Dataset_Classes)

'''
Digits scenario for large domain gap
'''
Private_Dataset_Name_List = args.Private_Dataset_Name_List
Private_Data_Total_Len_List = args.Private_Data_Total_Len_List
Private_Data_Len_List = args.Private_Data_Len_List
Public_Dataset_Name = args.Public_Dataset_Name


Save_File_Name = 'Generalization'+Scenario+'_'+Public_Dataset_Name+'.npy'

if __name__ =='__main__':
    logger = init_logs()
    print(Scenario+Public_Dataset_Name)
    logger.info("Random Seed and Server Config")
    seed = Seed
    np.random.seed(seed)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    device_ids = args.device_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    logger.info("Load Participants' Data and Model")
    net_list = init_nets(n_parties=N_Participants,nets_name_list=Private_Net_Name_List,num_classes=Output_Channel)
    test_dl_list = []
    for index in range(N_Participants):
        private_dataset_name = Private_Dataset_Name_List[index]
        private_dataset_dir = Dataset_Dir + private_dataset_name
        _, test_dl, _, _ = get_dataloader(dataset=private_dataset_name, datadir=private_dataset_dir,
                                          train_bs=TrainBatchSize,test_bs=TestBatchSize, dataidxs=None)
        test_dl_list.append(test_dl)
    test_accurcy_all_list = []
    for key, value in Evaluation_Idea_Dict.items():
        for item in value:
            Method_Name = key
            Ablation_Name = item
            Method_Path = Idea_Dir + Method_Name + '/'
            logger.info('Evaluate Method Generalization Performance: '+Method_Name+'-'+Ablation_Name)
            test_accuracy_list = []
            '''
            Evaluate Methods Accuracy
            '''
            for particiapnt_index in range(N_Participants):
                '''
                加载模型
                '''
                network = net_list[particiapnt_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                netname = Private_Net_Name_List[particiapnt_index]
                dataset_name = Private_Dataset_Name_List[particiapnt_index]
                network.load_state_dict(torch.load(Method_Path+'Model_Storage/' +Scenario+'/'+Ablation_Name+'/'+
                Public_Dataset_Name+'/'+ netname + '_' + str(particiapnt_index) + '_' +dataset_name+'.ckpt'))
                particiapnt_test_accuracy_list = evaluate_network_generalization(network=network, dataloader_list=test_dl_list,
                particiapnt_index=particiapnt_index,logger=logger)
                del particiapnt_test_accuracy_list[particiapnt_index]
                test_accuracy_list.append(round(mean(particiapnt_test_accuracy_list),2))
            print(test_accuracy_list)
            logger.info('The average Accuracy of models on the test images:' + str(mean(test_accuracy_list)))
            test_accurcy_all_list.append(test_accuracy_list)
            test_accurcy_all_array = np.array(test_accurcy_all_list)
            np.save(Save_File_Name, test_accurcy_all_array)  # 保存为.npy格式
    '''
    Evaluate Original
    '''
    test_accuracy_list = []
    for index in range(N_Participants):
        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Private_Net_Name_List[index]
        dataset_name = Private_Dataset_Name_List[index]
        '''
        加载模型
        '''
        network.load_state_dict(torch.load(Original_Path+'Model_Storage/'+ netname + '_' + str(index) + '_' +dataset_name+'.ckpt'))
        particiapnt_test_accuracy_list = evaluate_network_generalization(network=network, dataloader_list=test_dl_list,
        particiapnt_index=index,logger=logger)
        del particiapnt_test_accuracy_list[index]
        test_accuracy_list.append(round(mean(particiapnt_test_accuracy_list),2))
    logger.info('The average Accuracy of models on the test images:' + str(mean(test_accuracy_list)))
    test_accurcy_all_list.append(test_accuracy_list)
    test_accurcy_all_array = np.array(test_accurcy_all_list)
