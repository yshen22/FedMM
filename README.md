This repository contains the code accompanying the paper  "
FedMM: Saddle Point Optimization for Federated Adversarial Domain Adaptation" Paper[link](https://arxiv.org/pdf/2110.08477.pdf): 

![network structure](figfedmm.jpg  "Problem description")

#### Requirements to run the code:
---

1. Python 3.7
2. Tensorflow 1.14.0
3. numpy 1.20.3
4. tqdm

#### Download dataset:
---

Download mnistm data:
```
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
```
Preprocess mnistm dataset
```
python create_mnistm.py 
```

#### Experiments on Federated Domain Adaptation:
---
Usage for supervised training on source domain at Phase S_0:
```
python experiment.py -train_mod='sup_train' -SUP_EPOCHS=10 -adv_loss='MDD' -ckpt_path=$CHECKPOINT_SAVE_DIR  
```

Usage for continual adversarail domain adaptation using domain domain discriminators: 
```
python experiment.py -SUP_EPOCHS=10 -SR_DISC_EPOCHS=5 -DA_EPOCHS=100 -adv_loss='MDD' -ckpt_path=$CHECKPOINT_SAVE_DIR  
```

### Reference
---

```
@misc{2110.08477,
Author = {Yan Shen, Jian Du, Han Zhao, Benyu Zhang, Zhanghexuan Ji, Mingchen Gao},
Title = {FedMM: Saddle Point Optimization for Federated Adversarial Domain Adaptation},
Year = {2021},
Eprint = {arXiv:2110.08477},
}
