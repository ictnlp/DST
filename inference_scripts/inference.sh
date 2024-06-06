export CUDA_VISIBLE_DEVICES=0

TGT_FILE=model_path
MODELFILE=checkpoints/${TGT_FILE}
DATAFILE=data_file
THRESHOLD=infer_threshold
REFERENCE=ref.txt

python fairseq_cli/generate.py ${DATAFILE} --path ${MODELFILE} --batch-size 100 --beam 1 --remove-bpe --threshold ${THRESHOLD} --preRead 2 --soft-attention > decoding.txt
grep ^H decoding.txt | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > translation.txt
./multi-bleu.perl -lc ${ref.txt} < translation.txt