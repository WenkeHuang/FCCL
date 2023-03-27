import pandas as pd
import os
import numpy as np
import csv

path = './data/'
# scenario = 'fl_digits'  # fl_cifar10, fl_digits, fl_officehome
# scenarios_list = ['fl_officehome', 'fl_office31', 'fl_digits', 'fl_officecaltech']
scenarios_list = ['fl_digits'] # 'fl_officehome, fl_office31, fl_digits, fl_officecaltech'
structures_list = ['homogeneity']  # homogeneity,heterogeneity
public_data = 'pub_cifar100'  # pub_cifar100，pub_tyimagenet pub_fmnist pub_market1501
public_aug = 'weak'  # weak strong
public_len = 5000  # 5000 10000
column_mean_acc_list = ['method', 'paragroup'] + ['epoch' + str(i) for i in range(40)] + ['MEAN']
domain_list = ['intra', 'inter']


def load_mean_acc_list(structure_path, domain):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '':
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path + '/args.csv'
                    args_pd = pd.read_table(args_path, sep=",")
                    args_pd = args_pd.loc[:, args_pd.columns]
                    args_pub_dataset = args_pd['public_dataset'][0]
                    args_pub_aug = args_pd['pub_aug'][0]
                    args_pub_len = args_pd['public_len'][0]
                    if args_pub_dataset == public_data and args_pub_aug == public_aug and args_pub_len == public_len:
                        if len(os.listdir(para_path)) != 1:
                            data = pd.read_table(para_path + '/mean_' + domain + '.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values
                            mean_acc_value = np.mean(acc_value, axis=0)
                            mean_acc_value = mean_acc_value.tolist()
                            mean_acc_value = [round(item, 2) for item in mean_acc_value]
                            # max_acc_value = max(mean_acc_value)
                            last_acc_vale = mean_acc_value[-3:]
                            last_acc_vale = np.mean(last_acc_vale)
                            mean_acc_value.append(round(last_acc_vale, 3))
                            # mean_acc_value.append(max_acc_value)
                            acc_dict[experiment_index] = [model, para] + mean_acc_value
                            experiment_index += 1
    return acc_dict


def load_each_acc_list(structure_path, domain):
    acc_dict = {}
    experiment_index = 0
    for model in os.listdir(structure_path):
        if model != '':
            model_path = os.path.join(structure_path, model)
            if os.path.isdir(model_path):  # Check this path = path to folder
                for para in os.listdir(model_path):
                    para_path = os.path.join(model_path, para)
                    args_path = para_path + '/args.csv'
                    args_pd = pd.read_table(args_path, sep=",")
                    args_pd = args_pd.loc[:, args_pd.columns]
                    args_pub_dataset = args_pd['public_dataset'][0]
                    args_pub_aug = args_pd['pub_aug'][0]
                    args_pub_len = args_pd['public_len'][0]
                    if args_pub_dataset == public_data and args_pub_aug == public_aug and args_pub_len == public_len:
                        if len(os.listdir(para_path)) != 1:
                            data = pd.read_table(para_path + '/' + domain + '_accs.csv', sep=",")
                            data = data.loc[:, data.columns]
                            acc_value = data.values
                            times = int(len(acc_value) / n_participants)
                            mean_acc_value = []
                            for i in range(n_participants):
                                domain_acc_value = acc_value[[n_participants * j + i for j in range(times)]]
                                domain_mean_acc_value = np.mean(domain_acc_value, axis=0)
                                last_mean_acc_value = domain_mean_acc_value[-3:]
                                last_mean_acc_value = np.mean(last_mean_acc_value)
                                mean_acc_value.append(last_mean_acc_value)
                                # mean_acc_value.append(domain_mean_acc_value[-1])
                            mean_acc_value = [round(item, 2) for item in mean_acc_value]
                            mean_acc_value.append(np.mean(mean_acc_value))
                            acc_dict[experiment_index] = [model, para] + mean_acc_value
                            experiment_index += 1
    return acc_dict


if __name__ == '__main__':
    for _, scenario in enumerate(scenarios_list):
        if scenario == 'fl_office31':
            column_each_acc_list = ['method', 'paragroup', 'I', 'II', 'III', 'AVG']
            n_participants = 3
        else:
            column_each_acc_list = ['method', 'paragroup', 'I', 'II', 'III', 'IV', 'AVG']
            n_participants = 4
        print('**************************************************************')
        scenario_path = os.path.join(path, scenario)
        for _, structure in enumerate(structures_list):
            for _, domain in enumerate(domain_list):
                print('Scenario: ' + scenario + ' Structure: ' + structure + ' Domain: ' + domain + ' Public： ' + public_data + ' len: ' + str(public_len))
                structure_path = os.path.join(scenario_path, structure)
                mean_acc_dict = load_mean_acc_list(structure_path, domain)
                mean_df = pd.DataFrame(mean_acc_dict)
                mean_df = mean_df.T
                mean_df.columns = column_mean_acc_list
                print(mean_df)
                mean_df.to_excel(os.path.join(structure_path, domain + '_output.xls'), na_rep=True)
                each_acc_dict = load_each_acc_list(structure_path, domain)
                each_df = pd.DataFrame(each_acc_dict)
                each_df = each_df.T
                # pd.set_option('display.max_columns', None)
                each_df.columns = column_each_acc_list
                print(each_df)
                print('**************************************************************')
