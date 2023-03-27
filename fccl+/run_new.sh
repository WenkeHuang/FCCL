
#python main.py --model fedmd  --device_id 0 --dataset fl_digits --structure homogeneity --csv_log &
#python main.py --model fedmd  --device_id 1 --dataset fl_office31 --public_dataset pub_cifar100 --structure heterogeneity --csv_log &

#python main.py --model fcclsim  --device_id 0 --dataset fl_office31 --public_dataset pub_fmnist --structure heterogeneity --pub_aug weak --csv_log &
#python main.py --model fcclsim  --device_id 1 --dataset fl_office31 --public_dataset pub_fmnist --structure heterogeneity --pub_aug strong --csv_log &

#python main.py --model fcclsim  --device_id 2 --dataset fl_digits --public_dataset pub_fmnist --structure heterogeneity --pub_aug weak --csv_log &
#python main.py --model fcclsim  --device_id 4 --dataset fl_digits --public_dataset pub_fmnist --structure heterogeneity --pub_aug strong --csv_log &




#python main.py --model fedmd  --device_id 2 --dataset fl_officehome --structure heterogeneity --csv_log &
#python main.py --model fedmd  --device_id 3 --dataset fl_officecaltech --structure heterogeneity --csv_log &

#python main.py --model feddf  --device_id 2 --dataset fl_digits --structure homogeneity --csv_log &
#python main.py --model feddf  --device_id 3 --dataset fl_office31 --structure homogeneity --csv_log &
#python main.py --model moon  --device_id 4 --dataset fl_digits --structure homogeneity --csv_log &
#python main.py --model moon  --device_id 5 --dataset fl_office31 --structure homogeneity --csv_log &

#python main.py --model fedproc  --device_id 0 --dataset fl_digits --structure homogeneity --csv_log &
#python main.py --model fedproc  --device_id 1 --dataset fl_office31 --structure homogeneity --csv_log &
#
#python main.py --model scaffold  --device_id 2 --dataset fl_digits --structure homogeneity --csv_log &
#python main.py --model scaffold  --device_id 3 --dataset fl_office31 --structure homogeneity --csv_log &


#python main.py --model fedmd  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model feddf  --device_id 1 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fedmatch  --device_id 3 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model rhfl  --device_id 4 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fccl  --device_id 2 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fcclwe  --device_id 6 --dataset fl_digits --structure heterogeneity --csv_log &

#python main.py --model fcclwe  --device_id 3 --public_dataset pub_market1501 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fcclwe  --device_id 4 --public_dataset pub_market1501 --dataset fl_office31 --structure heterogeneity --csv_log &
#python main.py --model fcclwe  --device_id 5 --public_dataset pub_market1501 --dataset fl_officehome --structure heterogeneity --csv_log &
#python main.py --model fcclwe  --device_id 6 --public_dataset pub_market1501 --dataset fl_officecaltech --structure heterogeneity --csv_log &

### ewc e_lambda 0.01 0.1 1

#python main.py --model fcclewc  --device_id 1 --public_dataset pub_cifar100 --dataset fl_office31 --structure heterogeneity --csv_log &
#python main.py --model fcclewc  --device_id 1 --public_dataset pub_cifar100 --dataset fl_office31 --structure heterogeneity --csv_log &
#python main.py --model fcclewc  --device_id 3 --public_dataset pub_cifar100 --dataset fl_officehome --structure heterogeneity --csv_log &
#python main.py --model fcclewc  --device_id 4 --public_dataset pub_cifar100 --dataset fl_officecaltech --structure heterogeneity --csv_log &


#python main.py --model fedmd  --device_id 6 --dataset fl_officecaltech --structure heterogeneity --csv_log &
#python main.py --model feddf  --device_id 7 --dataset fl_officecaltech --structure heterogeneity --csv_log &
#python main.py --model fedmatch  --device_id 3 --dataset fl_officecaltech --structure heterogeneity --csv_log &
#python main.py --model rhfl  --device_id 2 --dataset fl_officecaltech --structure heterogeneity --csv_log &
#python main.py --model fccl  --device_id 5 --dataset fl_officecaltech --structure heterogeneity --csv_log &
#python main.py --model fcclwe  --device_id 4 --dataset fl_officecaltech --structure heterogeneity --csv_log &

#python main.py --model fcclwehalf  --device_id 0 --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fcclwehalf  --device_id 1 --dataset fl_office31 --structure heterogeneity --csv_log &

python main.py --model fedrs  --device_id 5 --public_dataset pub_cifar100 --dataset fl_digits --structure homogeneity --csv_log &
python main.py --model fedrs  --device_id 6 --public_dataset pub_cifar100 --dataset fl_office31 --structure homogeneity --csv_log &
python main.py --model fedproc  --device_id 3 --public_dataset pub_cifar100 --dataset fl_digits --structure homogeneity --csv_log &
python main.py --model fedproc  --device_id 7 --public_dataset pub_cifar100 --dataset fl_office31 --structure homogeneity --csv_log &

#python main.py --model feddf  --device_id 1 --public_dataset pub_tyimagenet --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model feddf  --device_id 2 --public_dataset pub_fmnist --dataset fl_digits --structure heterogeneity --csv_log &
#python main.py --model fedmd  --device_id 4 --public_dataset pub_cifar100 --dataset fl_officecaltech --structure heterogeneity --csv_log &
