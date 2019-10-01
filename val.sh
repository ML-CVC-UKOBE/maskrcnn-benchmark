FILE='configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml'
EXTRA='DATASETS.SUBMIT_ONLY True'
#EXTRA=' DATASETS.VISUALIZE True'
# EXTRA=''
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE TEST.IMS_PER_BATCH 4 OUTPUT_DIR 'output/x_152_8epochs' $EXTRA

