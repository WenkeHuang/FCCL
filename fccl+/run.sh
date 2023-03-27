#python main.py --model fcclsim  --device_id 7 --dataset fl_digits --structure heterogeneity --public_dataset pub_cifar100 --pub_aug weak --csv_log&
#python main.py --model fcclsim  --device_id 6 --dataset fl_digits --structure heterogeneity --public_dataset pub_cifar100 --pub_aug strong --csv_log&
#wait
#python main.py --model fcclsim  --device_id 7 --dataset fl_digits --structure heterogeneity --public_dataset pub_tyimagenet --pub_aug weak --csv_log&
#python main.py --model fcclsim  --device_id 6 --dataset fl_digits --structure heterogeneity --public_dataset pub_tyimagenet --pub_aug strong --csv_log&

#python main.py --model fcclwe --device_id 7 --dataset fl_officehome --structure heterogeneity --csv_log


#python main.py --model sgd --device_id 4 --dataset fl_digits --structure homogeneity --csv_log&

#python main.py --model sgd  --device_id 6 --dataset fl_digits --structure homogeneity --csv_log&
#python main.py --model fedavg  --device_id 6 --dataset fl_digits --structure homogeneity --csv_log&
#python main.py --model fedprox  --device_id 7 --dataset fl_digits --structure homogeneity --csv_log&
#python main.py --model moon  --device_id 7 --dataset fl_digits --structure homogeneity --csv_log&

#python main.py --model fcclwe  --device_id 6 --dataset fl_officehome --structure heterogeneity --csv_log&
#python main.py --model fcclkeyl  --device_id 7 --dataset fl_digits --structure homogeneity --csv_log
#python main.py --model fcclwe  --device_id 7 --dataset fl_digits --structure homogeneity --csv_log&
python main.py --model fccl  --device_id 0 --dataset fl_digits --structure homogeneity --csv_log&
python main.py --model fccl  --device_id 1 --dataset fl_office31 --structure homogeneity --csv_log&


#python main.py --model fedavg  --device_id 4 --dataset fl_office31 --structure homogeneity --csv_log&
#python main.py --model fedprox  --device_id 4 --dataset fl_office31 --structure homogeneity --csv_log&
#python main.py --model moon  --device_id 4 --dataset fl_office31 --structure homogeneity --csv_log&

#python main.py --model fedavg  --device_id 4 --dataset fl_digits --structure homogeneity --csv_log&
#python main.py --model fedprox  --device_id 4 --dataset fl_digits --structure homogeneity --csv_log&
#python main.py --model moon  --device_id 4 --dataset fl_digits --structure homogeneity --csv_log&

#python main.py --model fcclwe  --device_id 7 --dataset fl_office31 --structure heterogeneity --csv_log
#python main.py --model fccledg  --device_id 6 --dataset fl_digits --structure heterogeneity --csv_log
#
#python main.py --model fcclwe  --device_id 5 --dataset fl_digits --structure heterogeneity --csv_log
#python main.py --model fccl  --device_id 0 --dataset fl_officecaltech --structure heterogeneity --csv_log
#python main.py --model fedmd  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log
#python main.py --model feddf  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log
#wait
#python main.py --model fccledg  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fccltopsim  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log&

#python main.py --model fedmatch  --device_id 1 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fcclwe  --device_id 7 --dataset fl_officehomee --structure heterogeneity --csv_log&


#python main.py --model fccl  --device_id 2 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fcclwe  --device_id 3 --dataset fl_digits --structure heterogeneity --csv_log&
#wait
#python main.py --model fccledg  --device_id 2 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fcclgen  --device_id 3 --dataset fl_digits --structure heterogeneity --csv_log&



#python main.py --model feddf  --device_id 7 --dataset fl_officehome --structure heterogeneity --csv_log&
#python main.py --model fedmd  --device_id 7 --dataset fl_officehome --structure heterogeneity --csv_log&

#wait
#python main.py --model feddf  --device_id 3 --dataset fl_officecaltech --structure heterogeneity --csv_log
#python main.py --model fedmd  --device_id 3 --dataset fl_officecaltech --structure heterogeneity --csv_log
#python main.py --model fccl  --device_id 3 --dataset fl_officecaltech --structure heterogeneity --csv_log
#python main.py --model fcclwe  --device_id 3 --dataset fl_officehome --structure heterogeneity --csv_log



# fl_officehome
#python main.py --model fccl  --device_id 4 --dataset fl_officecaltech --structure heterogeneity --csv_log&
#python main.py --model wfccltopsim  --device_id 0 --dataset fl_officecaltech --structure heterogeneity --csv_log&
#wait
#python main.py --model wfccl  --device_id 4 --dataset fl_officecaltech --structure heterogeneity --csv_log&
#python main.py --model fccltopsim  --device_id 0 --dataset fl_officecaltech --structure heterogeneity --csv_log&
#wait
#python main.py --model fcclplus  --device_id 0 --dataset fl_officecaltech --structure heterogeneity --csv_log&
#
#wait

#python main.py --model fccl  --device_id 1 --dataset fl_office31 --structure heterogeneity --csv_log&
#python main.py --model wfccl  --device_id 2 --dataset fl_office31 --structure heterogeneity --csv_log&
#python main.py --model wfccltopsim  --device_id 2 --dataset fl_office31 --structure heterogeneity --csv_log&
#python main.py --model fccltopsim  --device_id 4 --dataset fl_office31 --structure heterogeneity --csv_log&
#python main.py --model fcclplus  --device_id 4 --dataset fl_office31 --structure heterogeneity --csv_log&


#python main.py --model feddf  --device_id 4 --dataset fl_digits --structure heterogeneity --csv_log&

#python main.py --model fccl  --device_id 1 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fcclplus  --device_id 2 --dataset fl_digits --structure heterogeneity --csv_log&
#python main.py --model fcclss  --device_id 3 --dataset fl_digits --structure heterogeneity --csv_log&



#python main.py --model fedavg  --device_id 0 --dataset fl_officehome --structure homogeneity --csv_log&
#python main.py --model fedprox  --device_id 1 --dataset fl_officehome --structure homogeneity --csv_log&
##python main.py --model moon  --device_id 3 --dataset fl_officehome --structure homogeneity --csv_log&
#python main.py --model fedmd  --device_id 2 --dataset fl_officehome --structure homogeneity --csv_log&
#python main.py --model feddf  --device_id 3 --dataset fl_officehome --structure homogeneity --csv_log&

#wait fl_cifar10 fl

#python main.py --model fedavg  --device_id 1 --dataset fl_digits --structure heterogeneity &
#python main.py --model fedprox  --device_id 4 --dataset fl_digits --structure heterogeneity &
#python main.py --model moon  --device_id 3 --dataset fl_digits --structure heterogeneity &
#python main.py --model fedmd  --device_id 5 --dataset fl_digits --structure heterogeneity &
#python main.py --model feddf  --device_id 6 --dataset fl_digits --structure heterogeneity &