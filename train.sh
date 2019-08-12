python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR 'output'
