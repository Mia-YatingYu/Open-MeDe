ROOT=/home/yuyating/workspace/OpenMeDe
CKPT=/data3/yuyating/CKPT

# expand + MAML_Debias
OUT_DIR=$CKPT/expand_FOMAML_virtual/vitb16_8x16/testing/ucf
#LOAD_CKPT_FILE=$CKPT/expand_FOMAML_virtual/vitb16_8x16/checkpoints/checkpoint_epoch_0022.pyth
LOAD_CKPT_FILE=$CKPT/expand_FOMAML_virtual/vitb16_8x16/wa_checkpoints/gwa_2_22.pth


cd $ROOT
export PYTHONPATH=$ROOT/slowfast:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TORCH_DISTRIBUTED_DEBUG=INFO python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/ucf101_full \
    DATA.PATH_PREFIX /data3/yuyating/dataset/OpenDataLab___UCF101/raw/videos_se256 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/ucf101-index2cls.json \
    DATA.TEXT_AUG False \
    DATA.TEXT_AUG_FEATURE_FILE $CKPT/text_feats/classes_feats_ucf101_tpl8_xmix_ViT-B_16.pt \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    OUTPUT_DIR $OUT_DIR \
    TRAIN.BATCH_SIZE 64 \
    TEST.BATCH_SIZE 256 \
    NUM_GPUS 4 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 101 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 1 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO 0. \
    MODEL.KEEP_RAW_MODEL True \
    MODEL.ENSEMBLE_PRED True \
    MODEL.ENSEMBLE_RAWMODEL_RATIO 0.3 \
    TEST.CLIP_ORI_PATH /home/yuyating/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 2 \
    MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
#    TENSORBOARD.ENABLE True \
#    TENSORBOARD.LOG_DIR $OUT_DIR/tensorboard \
#    TEST.CHECKPOINT_FILE_PATH $LOAD_CKPT_FILE \




