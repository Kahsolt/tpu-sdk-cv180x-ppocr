# PPOCR basic system with fuse proprocess and quant to INT8 & BF16

ℹ 32MB 的 ION Mem 无法做到同时加载多个模型 (尤其是 `ppocrv4_det_int8`)，因此不得不反复 load/unload 模型，额外开销巨大！！

⚪ det model (INT8)

| det | runnable? | quality |
| :-: | :-: | :-: |
| ppocrv4_det_int8  | √ | good, clear |
| ppocrv3_det_int8  | √ | has bad fragments |
| ppocrv2_det_int8  | √ | good, clear |
| ppocr_mb_det_int8 | √ | bad, missing areas |

⚪ rec model (BF16)

| model | runnable? | quality |
| :-: | :-: | :-: |
| ppocr_mb_rec_bf16  | √ | ? |


```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_sys
# upload cvimodel & runtime
scp ./cvimodels/ppocr*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
source ./envs_tpu_sdk.sh
cd samples
./bin/cvi_sample_ppocr_sys \
  ../cvimodels/ppocrv4_det_int8.cvimodel \
  ../cvimodels/ppocr_mb_rec_bf16.cvimodel \
	./data/gt_7148.jpg

./bin/cvi_sample_ppocr_sys ../cvimodels/ppocrv4_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/gt_97.jpg
./bin/cvi_sample_ppocr_sys ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/gt_97.jpg
./bin/cvi_sample_ppocr_sys ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/gt_97.jpg
./bin/cvi_sample_ppocr_sys ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/gt_97.jpg

# run on host (debug)
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/bitmap.png .
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/crop_box-*.png .
```
