FILE='configs/oid/e2e_faster_rcnn_mdconv_X-152-32x8d-FPN-IN5k_1.44x_caffe2.yaml'
NGPUS=8
sleep 5

for GROUP_S in $(seq 0 50 450)
do
    GROUP_E="$((GROUP_S+50))"
    EXTRA='DATASETS.SUBSET ("groupByCount_'$GROUP_S'_'$GROUP_E'",) MODEL.WEIGHT https://ucb456c1921ab4110d1ffc46be7e.dl.dropboxusercontent.com/cd/0/get/AoXyNneW6u3GNFK-8Ou-yLagbQrRe-iSB3H1Q8J7bRWp9zHQPgEwoB4MFvsM1wa_Uwkz20GzdYDC1DK5MK7YMeztdOEYgUQyrFG3aExmonA_TvmnoMv1Vp3NPaT-D10O8Fg/file# SOLVER.MAX_EPOCHS 12 SOLVER.SCHEDULER cosine SOLVER.ETA_MIN 0'
    python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE OUTPUT_DIR 'output/group_'$GROUP_S'_'$GROUP_E MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 $EXTRA
    sleep 5
done

for TOPIC in $(seq 0 10)
do
    EXTRA='DATASETS.SUBSET ("groupByTopic_10_'$TOPIC'",) MODEL.WEIGHT https://ucb456c1921ab4110d1ffc46be7e.dl.dropboxusercontent.com/cd/0/get/AoXyNneW6u3GNFK-8Ou-yLagbQrRe-iSB3H1Q8J7bRWp9zHQPgEwoB4MFvsM1wa_Uwkz20GzdYDC1DK5MK7YMeztdOEYgUQyrFG3aExmonA_TvmnoMv1Vp3NPaT-D10O8Fg/file# SOLVER.MAX_EPOCHS 1 SOLVER.SCHEDULER cosine SOLVER.ETA_MIN 0'
    python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file $FILE OUTPUT_DIR 'output/_topic_'$TOPIC MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN 2000 SOLVER.IMS_PER_BATCH 2 $EXTRA
    sleep 5
done

for GROUP_S in $(seq 0 50 450)
do
    GROUP_E="$((GROUP_S+50))"

    EXTRA='DATASETS.SUBMIT_ONLY True'
    python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE OUTPUT_DIR 'output/group_'$GROUP_S'_'$GROUP_E TEST.IMS_PER_BATCH 16 $EXTRA
    sleep 5
done

for TOPIC in $(seq 0 10)
do
    EXTRA='DATASETS.SUBMIT_ONLY True'
    python3 -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file $FILE OUTPUT_DIR 'output/_topic_'$TOPIC TEST.IMS_PER_BATCH 16 $EXTRA
    sleep 5
done

