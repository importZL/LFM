# Learning_From_Mistakes
Learning From Mistakes: A Multi-level Optimization Framework (Official Pytorch implementation for applications to Neural Architecture Search (NAS) and Data Reweighting (DR)).

## Setups
The requiring environment is as bellow:  

- Linux 
- Python 3+
- PyTorch 1.8.1 
- Torchvision 0.9.1

Or, you can use the following command to build the environment and get started:
```bash
conda env create -f environment.yml
```


## Running application to NAS on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example about running the search stage of DARTS on CIFAR-10:
```bash
python train_search_lfm.py --is_cifar100 0 --gpu 0 --unrolled --save darts-cifar10
```

Here is an example about running the evaluation stage of architecture searched on CIFAR-10:
```bash
python train.py --gpu 0 --auxiliary --cutout --arch [searched architecture]
```

## Running application to DR on benchmark datasets (CIFAR-10 and CIFAR-100).
Here is an example about running the experiment on class imbalance dataset with 100 imbalance factor
```bash
python dr-lfm-imbalance.py --dataset cifar100 --num_classes 100 --imb_factor 0.01
```

## Checkpoints that related to the results showed in the paper

Checkpoints of the Application to NAS:

- [Evaluation checkpoints for DARTS on CIFAR-10](https://drive.google.com/file/d/1cl4V1JojdQcByVShed5fp9iOvvfg9Qc7/view?usp=share_link)
- [Evaluation checkpoints for P-DARTS on CIFAR-10](https://drive.google.com/file/d/1nl6pmMX5BhwRtAAp0xdgPaEN7V7lEiCF/view?usp=share_link)
- [Evaluation checkpoints on ImageNet with architecture search by DARTS on CIFAR-10](https://drive.google.com/file/d/1LuIJqDy87pB03NpnNMllZwzTjAizR4tW/view?usp=share_link)
- [Evaluation checkpoints on ImageNet with architecture search by P-DARTS on CIFAR-10](https://drive.google.com/file/d/1yj4YGFI5-L6EeVkb1jUD21DsG4aXI5dK/view?usp=share_link)

Checkpoints of the Application to DR (Class Imbalance):

- [Checkpoints for experiment on CIFAR-10 with 10 Imbalance Factor](https://drive.google.com/file/d/1ejRH94NsHPU8PHw_npokAbMgeBU9YoBm/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-10 with 20 Imbalance Factor](https://drive.google.com/file/d/1cl4V1JojdQcByVShed5fp9iOvvfg9Qc7/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-10 with 50 Imbalance Factor](https://drive.google.com/file/d/18hVcsOLkqOVj9gi_i_Fjn0SKExfUW-3r/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-10 with 100 Imbalance Factor](https://drive.google.com/file/d/1EMJQDhd1k9Ld5UXUhysQGIpmFeX4C9j_/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-10 with 200 Imbalance Factor](https://drive.google.com/file/d/1hCk7waLtFcX9aQepXOctf3hx4sGtvEKY/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 10 Imbalance Factor](https://drive.google.com/file/d/11-rfWYaiabWIF9V6TTBkZSHnEjxOMedc/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 20 Imbalance Factor](https://drive.google.com/file/d/1dRlPT1zFUq0Ywvq0DnruwYTkjtbWvCnS/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 50 Imbalance Factor](https://drive.google.com/file/d/1sI0e48nTKwz6A3U9jjQorDDPrpXsFa5w/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 100 Imbalance Factor](https://drive.google.com/file/d/1s61tttllb-FyL73NegiptU76vMzPeKtq/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 200 Imbalance Factor](https://drive.google.com/file/d/12SmjnJW2fEWchtGrxGPoisk18T_hhvMH/view?usp=share_link)

Checkpoints of the Application to DR (Label Noisy):

- [Checkpoints for experiment on CIFAR-10 with 0.2 Flip Noisy](https://drive.google.com/file/d/1pWbMvYcmQh5sNTeJNhX2PuYL_fYJw3iO/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-10 with 0.4 Flip Noisy](https://drive.google.com/file/d/1ypU7iGTW1-4jS15ucXuZk_kwsbhpw_h6/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 0.2 Flip Noisy](https://drive.google.com/file/d/1Ke1t7GYZ5sQT0seVW1L6R5n7qcU3rjex/view?usp=share_link)
- [Checkpoints for experiment on CIFAR-100 with 0.4 Flip Noisy](https://drive.google.com/file/d/1QAQeBFrBcPpCaUuiCit5SSesYzWAZMc0/view?usp=share_link)

