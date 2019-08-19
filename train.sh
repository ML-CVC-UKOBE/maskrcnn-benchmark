FILE=configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml
EXTRA='DATASETS.SUBSET ("groupByCount_0_10",) MODEL.WEIGHT e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-OID.pth'
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR 'output' $EXTRA
