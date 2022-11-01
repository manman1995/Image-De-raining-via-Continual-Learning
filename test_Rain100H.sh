#!/bin/bash

# PReNet
python test_PReNet.py --test_model /gdata/xiaojie/Prenet_model/0/Rain100H_net_epoch_100.pth --save_path /gdata/xiaojie/PReNet_results/1_Real --data_path /gdata/xiaojie/Prenet_Dataset/test/Real_Internet --prefix SH
python test_PReNet.py --test_model /gdata/xiaojie/Prenet_model/0/Rain100L_net_epoch_100.pth --save_path /gdata/xiaojie/PReNet_results/1_Real --data_path /gdata/xiaojie/Prenet_Dataset/test/Real_Internet --prefix SL
python test_PReNet.py --test_model /gdata/xiaojie/Prenet_model/001/Rain100L_net_epoch_100.pth --save_path /gdata/xiaojie/PReNet_results/1_Real --data_path /gdata/xiaojie/Prenet_Dataset/test/Real_Internet --prefix PIG
#python test_PReNet.py --test_model /gdata/xiaojie/PRN_model/1/Rain100L_net_epoch_100.pth --save_path /gdata/xiaojie/PRN_results/1_Rain100L --data_path /gdata/xiaojie/Prenet_Dataset/test/Rain100L/rainy --prefix L
#python test_PReNet.py --test_model /gdata/xiaojie/PRN_model/1/Rain100L_net_epoch_100.pth --save_path /gdata/xiaojie/PRN_results/1_Rain100L --data_path /gdata/xiaojie/Prenet_Dataset/test/Rain100H/rainy --prefix H

