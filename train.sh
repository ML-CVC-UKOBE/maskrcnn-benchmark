FILE='configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml'
NGPUS=2

for GROUP_S in $(seq 0 50 450)
do
    GROUP_E="$((GROUP_S+50))"
    /bin/sh -c " python -m torch.distributed.launch --nproc_per_node=8 tools/train_net.py --config-file $FILE OUTPUT_DIR output/group_${GROUP_S}_${GROUP_E} MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 16 DATASETS.SUBSET \"('groupByCount_${GROUP_S}_${GROUP_E}',)\" MODEL.WEIGHT https://uc3fac17cc438db45141ed265d59.dl.dropboxusercontent.com/cd/0/get/AoqmyC99x-SDAExZ4vAGz0-NoUr_9ruCUg-DmS2cDLLwpqXMoX-fsD24FubUoye7lHHwefWfc9h_74TaFhcGxpUlwWFg3Z_e-YfgI-R8j2ZYPueAUZQoV1EbG8hfINWbP6w/file\# SOLVER.MAX_EPOCHS 12 SOLVER.SCHEDULER cosine SOLVER.ETA_MIN 0"
EXTRA='DATASETS.SUBMIT_ONLY True'
python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE TEST.IMS_PER_BATCH 4 OUTPUT_DIR 'output/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2/oid_v5_challenge_train_expanded/2019-08-14_18:47' $EXTRA
done

