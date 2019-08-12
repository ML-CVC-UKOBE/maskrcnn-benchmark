FILE=configs/oid/e2e_faster_rcnn_mdconv_R_50_FPN_1x.yaml
#python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE TEST.IMS_PER_BATCH 16 OUTPUT_DIR $OUTPUT 
python3 tools/test_net.py --config-file $FILE TEST.IMS_PER_BATCH 16 OUTPUT_DIR $OUTPUT 
