# Learn from others and Be yourself in Heterogeneous Federated Learning

> Rethinking Federated Learning with Domain Shift: A Prototype View,            
> Wenke Huang, Mang Ye, Bo Du
> *CVPR, 2022*
> [Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)

## Abstract
Federated learning has emerged as an important dis- tributed learning paradigm, which normally involves col- laborative updating with others and local updating on pri- vate data. However, heterogeneity problem and catas- trophic forgetting bring distinctive challenges. First, due to non-i.i.d (identically and independently distributed) data and heterogeneous architectures, models suffer perfor- mance degradation on other domains and communication barrier with participants models. Second, in local updat- ing, model is separately optimized on private data, which is prone to overfit current data distribution and forgets pre- viously acquired knowledge, resulting in catastrophic for- getting. In this work, we propose FCCL (Federated Cross- Correlation and Continual Learning). For heterogeneity problem, FCCL leverages unlabeled public data for com- munication and construct cross-correlation matrix to learn a generalizable representation under domain shift. Mean- while, for catastrophic forgetting, FCCL utilizes knowledge distillation in local updating, providing inter and intra do- main information without leaking privacy. Empirical re- sults on various image classification tasks demonstrate the effectiveness of our method and the efficiency of modules.

## Citation
```
@inproceedings{FCCL_CVPR22,
    title={Learn from others and be yourself in heterogeneous federated learning},
    author={Huang, Wenke and Ye, Mang and Du, Bo},
    booktitle={CVPR},
    year={2022}
}
```

## Relevant Projects
[1] Rethinking Federated Learning with Domain Shift: A Prototype View
 - CVPR 2023 [Code](https://github.com/WenkeHuang/RethinkFL)]
