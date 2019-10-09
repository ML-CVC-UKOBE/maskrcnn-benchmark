FILE='configs/lvis/e2e_mask_rcnn_R_50_FPN_1x.yaml'
# FILE='configs/e2e_faster_rcnn_R_50_FPN_1x.yaml'
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE SOLVER.IMS_PER_BATCH 4 OUTPUT_DIR 'output'
