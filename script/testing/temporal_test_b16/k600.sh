ROOT=/home/yuyating/workspace/OpenMeDe
CKPT=/data3/yuyating/CKPT

# expand + MAML_Debias
OUT_DIR=$CKPT/expand_FOMAML_virtual/vitb16_8x16/testing/ucf
#LOAD_CKPT_FILE=$CKPT/expand_FOMAML_virtual/vitb16_8x16/checkpoints/checkpoint_epoch_0022.pyth
LOAD_CKPT_FILE=$CKPT/expand_FOMAML_virtual/vitb16_8x16/wa_checkpoints/gwa_2_22.pth

K600_split=k600_split1
PATCHING_RATIO=0.5

cd $ROOT
export PYTHONPATH=$ROOT/slowfast:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1 \
python -W ignore -u tools/run_net.py \
    --cfg configs/Kinetics/TemporalCLIP_vitb16_8x16_STAdapter.yaml \
    --opts DATA.PATH_TO_DATA_DIR $ROOT/zs_label_db/$K600_split \
    DATA.PATH_PREFIX /data3/yuyating/dataset/Kinetics-600/videos_se256 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE $ROOT/zs_label_db/$K600_split/k600-openset-index2cls.json \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 128 \
    NUM_GPUS 2 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 160 \
    MODEL.TEMPORAL_MODELING_TYPE 'expand_temporal_view' \
    MODEL.KEEP_RAW_MODEL True \
    MODEL.ENSEMBLE_PRED True \
    MODEL.ENSEMBLE_RAWMODEL_RATIO 0.3 \
    TEST.CUSTOM_LOAD False \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.PATCHING_RATIO $PATCHING_RATIO \
    TEST.CLIP_ORI_PATH /home/yuyating/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 2 \


