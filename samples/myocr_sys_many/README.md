# My mixed OCR system with fuse proprocess and quant to INT8 & BF16

一次性推理整个文件夹！其余请参考 myocr_sys 子项目 ;)

```shell
# compile runtime
bash ./compile_sample_runner.sh myocr_sys_many
# upload cvimodel & runtime
scp ./cvimodels/*ocr*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
# upload infer data
ssh root@192.168.42.1 "mkdir -p /dataset/train_full_images_0"
scp /path/to/img/*.jpg root@192.168.42.1:/dataset/train_full_images_0

# run on chip
source ./envs_tpu_sdk.sh
cd samples
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
```
