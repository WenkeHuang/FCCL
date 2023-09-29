# Learn from others and Be yourself in Heterogeneous Federated Learning

> Learn from others and Be yourself in Heterogeneous Federated Learning,            
> Wenke Huang, Mang Ye, Bo Du
> *CVPR, 2022*
> [Link](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_Learn_From_Others_and_Be_Yourself_in_Heterogeneous_Federated_Learning_CVPR_2022_paper.pdf)

## Abstract
Federated learning has emerged as an important distributed learning paradigm, which normally involves collaborative updating with others and local updating on private data. However, heterogeneity problem and catastrophic forgetting bring distinctive challenges. First, due to non-i.i.d (identically and independently distributed) data and heterogeneous architectures, models suffer performance degradation on other domains and communication barrier with participants models. Second, in local updating, model is separately optimized on private data, which is prone to overfit current data distribution and forgets previously acquired knowledge, resulting in catastrophic forgetting. In this work, we propose FCCL (Federated Cross-Correlation and Continual Learning). For heterogeneity problem, FCCL leverages unlabeled public data for communication and construct cross-correlation matrix to learn a generalizable representation under domain shift. Meanwhile, for catastrophic forgetting, FCCL utilizes knowledge distillation in local updating, providing inter and intra domain information without leaking privacy. Empirical results on various image classification tasks demonstrate the effectiveness of our method and the efficiency of modules.

## Citation
```
@inproceedings{FCCL_CVPR22,
    title={Learn from others and be yourself in heterogeneous federated learning},
    author={Huang, Wenke and Ye, Mang and Du, Bo},
    booktitle={CVPR},
    year={2022}
}
@misc{FCCLPlus_arXiv23,
      title={Generalizable Heterogeneous Federated Cross-Correlation and Instance Similarity Learning}, 
      author={Wenke Huang and Mang Ye and Zekun Shi and Bo Du},
      year={2023},
      eprint={2309.16286},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Relevant Projects
[1] Rethinking Federated Learning with Domain Shift: A Prototype View - CVPR 2023 [[Link](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf)] [[Code](https://github.com/WenkeHuang/RethinkFL)]

[2] Federated Graph Semantic and Structural Learning - IJCAI 2023 [[Link](https://marswhu.github.io/publications/files/FGSSL.pdf)][[Code](https://github.com/wgc-research/fgssl)]
