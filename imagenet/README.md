# OWTTT on ImageNet-C/R

Ours method and the baseline method TEST (direct test without adaptation) on ImageNet-C/ImageNet-R under the OWTTT protocol. Our implementation is based on [repo](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/imagenet) and therefore requires some similar preparation processes.

### Requirements

- To install requirements:

    ```
    pip install -r requirements.txt
    ```

- To download ImageNet dataset:

    We need to firstly download the validation set and the development kit (Task 1 & 2) of ImageNet-1k on [here](https://image-net.org/challenges/LSVRC/2012/index.php), and put them under `data` folder.

- To download ImageNet-R dataset:

    To download datasets:

    ```
    export DATADIR=/data
    cd ${DATADIR}
    wget -O imagenet-r.tar https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    tar -xvf imagenet-r.tar
    ```

- To create the corruption dataset
    ```
    python utils/create_corruption_dataset.py
    ```

    The issue `Frost missing after pip install` can be solved following [here](https://github.com/hendrycks/robustness/issues/4#issuecomment-427226016).

    Finally, the structure of the `data` folder should be like
    ```
    data
    |_ ILSVRC2012_devkit_t12.tar
    |_ ILSVRC2012_img_val.tar
    |_ val
        |_ n01440764
        |_ ...
    |_ imagenet-r
        |_ n01443537
        |_ ...
    |_ corruption
        |_ brightness.pth
        |_ contrast.pth
        |_ ...
    |_ meta.bin
    ```

### Pre-trained Models

Here, we use the pretrain model provided by torchvision.

### Open-World Test-Time Training:

We present our method and the baseline method TEST (direct test without adaptation) on ImageNet-C/R.

- run OURS method or the baseline method TEST on ImageNet-C under the OWTTT protocol.

    ```
    # OURS: 
    bash scripts/ours_c.sh "corruption_type" "strong_ood_type" 

    # TEST: 
    bash scripts/test_c.sh "corruption_type" "strong_ood_type" 
    ```
    Where "corruption_type" is the corruption type in ImageNet-C, and "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN]. 
    
    For example, to run OURS or TEST on ImageNet-C under the snow corruption with MNIST as strong OOD, we can use the following command:

    ```
    # OURS:
    bash scripts/ours_c.sh snow MNIST 

    # TEST:
    bash scripts/test_c.sh snow MNIST
    ```

    The following results are yielded by the above scripts (%) under the snow corruption, and with MNIST as strong OOD:

    | Method | ACC_S | ACC_N | ACC_H |
    |:------:|:-------:|:-------:|:-------:|
    |  TEST  |   17.30   |    99.35   |   29.47  |
    |  OURS  |   45.34    |    100.00   | 62.39 |

- run OURS method or the baseline method TEST on ImageNet-R under the OWTTT protocol.
    
    ```
    # OURS: 
    bash scripts/ours_cifar100.sh "strong_ood_type" 

    # TEST: 
    bash scripts/test_cifar100.sh "strong_ood_type" 
    ```
    Where "strong_ood_type" is the strong OOD type in [noise, MNIST, SVHN]. 
    
    For example, to run OURS or TEST on ImageNet-R with MNIST as strong OOD, we can use the following command:

    ```
    # OURS:
    bash scripts/ours_r.sh MNIST 

    # TEST:
    bash scripts/test_r.sh MNIST
    ```

    The following results are yielded by the above scripts (%) with MNIST as strong OOD:

    | Method | ACC_S | ACC_N | ACC_H |
    |:------:|:-------:|:-------:|:-------:|
    |  TEST  |   35.50   |    99.96   |   52.39  |
    |  OURS  |   41.40   |    100.00   |   58.56  |


### Acknowledgements

Our code is built upon the public code of the [TTAC](https://github.com/Gorilla-Lab-SCUT/TTAC/tree/master/imagenet).
