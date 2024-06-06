export CUDA_VISIBLE_DEVICES=2
CKPT_DIR=/data/guoshoutao/decoder_only_languageModel/checkpoints/adaptive_mergeweight/L3_T6_MASS0.9_0.4curriculum_40000up/average
a=0.95
b=0.05
	#echo -e "\n\n\n\nCheckpoint${j}"
	#checkpoint="checkpoints/"${dir}"/checkpoint${j}.pt"
	#echo -e "\n\n\n\n$file"

a=0.9
for i in {1..20}
do

    echo "soft-threshold"${a}"preRead2" >> ${CKPT_DIR}/result_top
    python /data/guoshoutao/decoder_only_languageModel/adaptive_attn_deltaProb_mergedWeight_curriculum/fairseq_cli/generate.py /data/guoshoutao/wmt15_de_en_bpe32k --path ${CKPT_DIR}/average-model.pt --batch-size 100 --beam 1 --remove-bpe --threshold ${a} --preRead 2 --soft-attention > ${CKPT_DIR}/ress_mt_${a} 2>&1
    tail -n 5 ${CKPT_DIR}/ress_mt_${a} >>  ${CKPT_DIR}/result_top
    grep ^H ${CKPT_DIR}/ress_mt_${a} | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${CKPT_DIR}/res.en
    /data/guoshoutao/multi-bleu.perl -lc  /data/guoshoutao/newstest2015.tok.en < ${CKPT_DIR}/res.en >> ${CKPT_DIR}/result_top

    a=$(echo "$a $b" | awk '{print $1-$2}')
done
