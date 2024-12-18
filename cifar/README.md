# OWTTT on CIFAR10-C/100-C

Ours method and the baseline method TEST (direct test without adaptation) on CIFAR-10-C/100-C under common corruptions or natural shifts. Our implementation is based on [repo](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/cifar) and therefore requires some similar preparation processes.


### Requirements

To install requirements:

```
pip install -r requirements.txt
```

To download datasets:

```
export DATADIR=/data/cifar
mkdir -p ${DATADIR} && cd ${DATADIR}
wget -O CIFAR-10-C.tar https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
tar -xvf CIFAR-10-C.tar
wget -O CIFAR-100-C.tar https://zenodo.org/record/3555552/files/CIFAR-100-C.tar?download=1
tar -xvf CIFAR-100-C.tar
wget -O tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip
```

### Pre-trained Models

The checkpoints of pre-train Resnet-50 can be downloaded (214MB) using the following command:

```
mkdir -p results/cifar10_joint_resnet50 && cd results/cifar10_joint_resnet50
gdown https://drive.google.com/uc?id=1QWyI8UrXJ6_H9lBbrq52qXWpjdpq4PUn && cd ../..
mkdir -p results/cifar100_joint_resnet50 && cd results/cifar100_joint_resnet50
gdown https://drive.google.com/uc?id=1cau93HVjl4aWuZlrl7cJIMEKBxXXunR9 && cd ../..
```

These models are obtained by training on the clean CIFAR10/100 images using semi-supervised SimCLR.

### Open-World Test-Time Training:

We present our method and the baseline method TEST (direct test without adaptation) on CIFAR10-C/100-C.

- run OURS method or the baeline method TEST on CIFAR10-C under the OWTTT protocol.

    ```
    # OURS: 
    bash scripts/ours_cifar10.sh "corruption_type" "strong_ood_type" 

    # TEST: 
    bash scripts/test_cifar10.sh "corruption_type" "strong_ood_type" 
    ```
    Where "corruption_type" is the corruption type in CIFAR10-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN, Tiny, cifar100]. 
    
    For example, to run OURS or TEST on CIFAR10-C under the snow corruption with MNIST as strong OOD, we can use the following command:

    ```
    # OURS:
    bash scripts/ours_cifar10.sh snow MNIST 

    # TEST:
    bash scripts/test_cifar10.sh snow MNIST
    ```

    The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:

    | Method | ACC_S | ACC_N | ACC_H |
    |:------:|:-------:|:-------:|:-------:|
    |  TEST  |   66.36   |    91.56   |   76.95  |
    |  OURS  |   84.05    |    97.46   | 90.26|

- run OURS method or the baeline method TEST on CIFAR100-C under the OWTTT protocol.
    
    ```
    # OURS: 
    bash scripts/ours_cifar100.sh "corruption_type" "strong_ood_type" 

    # TEST: 
    bash scripts/test_cifar100.sh "corruption_type" "strong_ood_type" 
    ```
    Where "corruption_type" is the corruption type in CIFAR100-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN, Tiny, cifar10]. 
    
    For example, to run OURS or TEST on CIFAR100-C under the snow corruption with MNIST as strong OOD, we can use the following command:

    ```
    # OURS:
    bash scripts/ours_cifar100.sh snow MNIST 

    # TEST:
    bash scripts/test_cifar100.sh snow MNIST
    ```

    The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:

    | Method | ACC_S | ACC_N | ACC_H |
    |:------:|:-------:|:-------:|:-------:|
    |  TEST  |   29.2   |    53.27   |   37.72  |
    |  OURS  |   44.78    |    93.56   |  60.57 |


### Acknowledgements

Our code is built upon the public code of the [TTAC](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/cifar).
