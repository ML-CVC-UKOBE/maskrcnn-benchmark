FILE='configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml'
EXTRA='DATASETS.SUBMIT_ONLY True'
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE TEST.IMS_PER_BATCH 4 OUTPUT_DIR 'output/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2/oid_v5_challenge_train_expanded/2019-08-14_18:47' $EXTRA

for GROUP_S in $(seq 0 50 400)
do
    GROUP_E="$((GROUP_S+50))"
    /bin/sh -c " python -m torch.distributed.launch --nproc_per_node=8 tools/test_net.py --config-file $FILE OUTPUT_DIR output/group_${GROUP_S}_${GROUP_E} TEST.IMS_PER_BATCH 32 DATASETS.SUBMIT_ONLY True"
done
