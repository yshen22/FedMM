create mnistm data:\\
curl -L -O http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz\\
python create_mnistm.py \\
FedMM on DANN loss: \\
python train.py -max_iter=15000 -lambda1_decay=1.05 -adv_loss='DANN' \\
FedMM on MDD loss: \\
python train.py -max_iter=50000 -lambda1_decay=1.01 -adv_loss='MDD' \\
FedMM on CDAN loss \\
python train.py -max_iter=30000 -lambda1_decay=1.02 -adv_loss='CDAN'
