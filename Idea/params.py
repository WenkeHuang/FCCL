import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    '''
    Default Setting
    '''
    parser.add_argument('--Dataset_Dir',type=str,default='')
    parser.add_argument('--Project_Dir',type=str,default='')
    parser.add_argument('--Original_Path',type=str,default='')
    parser.add_argument('--CommunicationEpoch',type=int,default=40)
    parser.add_argument('--Seed',type=int,default=42)
    parser.add_argument('--device_ids',type=list,default=[0,1])

    '''
    General Setting
    '''
    parser.add_argument('--N_Participants',type=int,default=4)
    '''
    Scenario for domain gap and corresponding public data parameteres
    '''
    Scenario='Digits'
    parser.add_argument('--Scenario',type=str,default=Scenario)
    parser.add_argument('--Public_Dataset_Name', type=str, default='cifar_100')

    if Scenario =='Digits':
        parser.add_argument('--Private_Net_Name_List', type=list,
                            default=['ResNet10', 'ResNet12', 'Efficientnet', 'Mobilenetv2'])
        parser.add_argument('--Private_Dataset_Name_List',type=list,default=['mnist', 'usps', 'svhn', 'syn'])
        parser.add_argument('--Private_Data_Total_Len_List',type=list,default=[60000, 7291, 73257, 10000])
        parser.add_argument('--Private_Data_Len_List',type=list,default=[150, 80, 5000, 1800])
        parser.add_argument('--Private_Training_Epoch',type=list,default=[40,35,3,4])
        parser.add_argument('--Private_Dataset_Classes',type=list,default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        parser.add_argument("--Private_Net_Feature_Dim_List",type =list,default=[512,512,320,1280])
        parser.add_argument('--TrainBatchSize',type=int,default=512)
        parser.add_argument('--Local_TrainBatchSize',type=int,default=256)
        parser.add_argument('--TestBatchSize',type=int,default=512)
        parser.add_argument('--Public_Training_Epoch',type=int,default=1)
        parser.add_argument('--Public_Dataset_Length',type=int,default=5000)

    if Scenario =='OfficeHome':

        parser.add_argument('--Private_Net_Name_List', type=list,
                            default=['ResNet18', 'ResNet18', 'ResNet18', 'ResNet18'])
        parser.add_argument('--Private_Dataset_Name_List',type=list,default=['Art', 'Clipart','Product','Real World'])
        parser.add_argument('--Private_Data_Total_Len_List',type=list,default=[1700,3050,3100,3050])
        parser.add_argument('--Private_Data_Len_List',type=list,default=[1400, 2000,2500,2000])
        parser.add_argument('--Private_Training_Epoch',type=list,default=[10,6,6,6])
        parser.add_argument('--Private_Dataset_Classes',type=list,default=['Alarm', 'Clock', 'Backpack', 'Batteries', 'Bed', 'Bike',
        'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles','Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk Lamp',
        'Drill', 'Eraser', 'Exit Sign', 'Fan','File Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet',
        'Kettle', 'Keyboard','Knives', 'Lamp Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan',
        'Paper Clip', 'Pen', 'Pencil', 'Postit Notes', 'Printer', 'Push Pin', 'Radio', 'Refrigerator', 'ruler','Scissors', 'Screwdriver',
        'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone','Toothbrush', 'Toys', 'Trash Can', 'TV', 'Webcam'])
        parser.add_argument('--TrainBatchSize',type=int,default=512)
        parser.add_argument('--TestBatchSize',type=int,default=256)
        parser.add_argument('--Public_Training_Epoch',type=int,default=2)
        parser.add_argument('--Public_Dataset_Length',type=int,default=5000)


    args = parser.parse_args()
    return args