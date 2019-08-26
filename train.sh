#FILE=configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml
FILE=configs/oid/e2e_faster_rcnn_R_50_FPN_1x.yaml
#FILE=configs/e2e_faster_rcnn_R_50_FPN_1x.yaml
#EXTRA='DATASETS.SUBSET ("groupByCount_0_100",) MODEL.WEIGHT e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-OID.pth'
EXTRA='DATASETS.SUBSET ("groupByCount_0_100",) OUTPUT_DIR_SUFFIX _ram_test_1'
#EXTRA='OUTPUT_DIR_SUFFIX _coco_test_1'
#EXTRA='DATASETS.SUBSET ("groupByCount_0_100",) MODEL.WEIGHT e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-OID.pth SOLVER.MAX_EPOCHS 5 SOLVER.SCHEDULER cosine SOLVER.ETA_MIN 0'
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR 'output' $EXTRA
#python3 tools/train_net.py --config-file $FILE MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 OUTPUT_DIR 'output' $EXTRA
