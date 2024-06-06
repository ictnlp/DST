export CUDA_VISIBLE_DEVICES=0,1,2,3

TRAIN_THRESHOLD=0.45
CURR_STEPS=4000
TGT_FILE=ckpt_${TRAIN_THRESHOLD}_${CURR_STEPS}
MODELFILE=checkpoints/${TGT_FILE}
DATAFILE=data_file

python train.py --ddp-backend=no_c10d ${DATAFILE} --arch transformer_lm \
 --task translation \
 --optimizer adam \
 --adam-betas '(0.9, 0.98)' \
 --clip-norm 0.0 \
 --lr 5e-4 \
 --lr-scheduler inverse_sqrt \
 --warmup-init-lr 1e-07 \
 --warmup-updates 4000 \
 --dropout 0.3 \
 --decoder-attention-heads 8 \
 --decoder-layers 16 \
 --criterion dst_loss \
 --report-accuracy \
 --label-smoothing 0.1 \
 --save-dir ${MODELFILE} \
 --max-tokens 2048 --update-freq 4 \
 --skip-invalid-size-inputs-valid-test \
 --keep-best-checkpoints 10 \
 --best-checkpoint-metric loss \
 --max-target-positions 1024 \
 --tokens-per-sample 1024 \
 --reset-optimizer \
 --reset-lr-scheduler \
 --reset-meters \
 --reset-dataloader \
 --train-threshold ${TRAIN_THRESHOLD} \
 --curr-steps ${CURR_STEPS} \
 --log-interval 100 > train_log/${TGT_FILE} 2>&1 &
