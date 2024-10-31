# My mixed OCR system with fuse proprocess and quant to INT8 & BF16

一次性推理整个文件夹！其余请参考 myocr_sys 子项目 ;)

⚠ chocr_rec 确实不支持竖排文本，认命。

性能优化建议:

  - 使用更小的 det 模型
  - 设置更大的 `DET_MIN_SIZE`
  - 不启用 `DEBUG_DUMP_*` 

使用自定义的 cvimodel 模型：

  - 检查 `DET_IMG_SIZE`, `REC_IMG_WIDTH`, `REC_IMG_HEIGHT` 设置是否与模型编译规格一致
  - 检查 `CVI_NN_TensorPtr` 返回值类型是否与模型编译规格一致

```shell
# compile runtime
bash ./compile_sample_runner.sh myocr_sys_many
# upload cvimodel & runtime
scp ./cvimodels/*ocr*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
# upload infer data
ssh root@192.168.42.1 "mkdir -p /dataset/train_full_images_0"
scp /path/to/img/gt_100*.jpg root@192.168.42.1:/dataset/train_full_images_0   # 19 images

# run on chip
source ./envs_tpu_sdk.sh
cd samples
# run with the highest priority! :)
nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv4_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0   # OOM
nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0

# run on host
scp -r root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results .
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .
python convert_results.py ./results.txt
```

⚪ ppocrv3_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocrv3_det Build at 2024-10-31 18:52:24 For platform cv180x
Max SharedMem size:8793600
version: 1.4.0
chocr_rec Build at 2024-10-31 18:05:38 For platform cv180x
Max SharedMem size:908000
find shared memory(8793600),  saved:908000 
ts_model_load: 860.000 ms
>> progress: ...................
ts_model_unload: 92.000 ms
================================
n_img:        19
n_crop:       40
--------------------------------
ts_img_load:  3598.220 ms
ts_img_crop:  317.727 ms
ts_det_pre:   1510.667 ms
ts_det_infer: 4605.342 ms
ts_det_post:  392.563 ms
ts_rec_pre:   64.259 ms
ts_rec_infer: 912.839 ms
ts_rec_post:  565.021 ms
--------------------------------
ts_avg_pre:   82.891 ms
ts_avg_infer: 290.431 ms
ts_avg_post:  50.399 ms
================================
Total time:   12968.000 ms
```

⚪ ppocrv2_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-10-31 18:27:55 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
chocr_rec Build at 2024-10-31 18:05:38 For platform cv180x
Max SharedMem size:908000
find shared memory(8179200),  saved:908000 
ts_model_load: 812.000 ms
>> progress: ...................
ts_model_unload: 92.000 ms
================================
n_img:        19
n_crop:       50
--------------------------------
ts_img_load:  3550.438 ms
ts_img_crop:  310.863 ms
ts_det_pre:   1484.322 ms
ts_det_infer: 4210.566 ms
ts_det_post:  316.821 ms
ts_rec_pre:   75.400 ms
ts_rec_infer: 1138.814 ms
ts_rec_post:  706.176 ms
--------------------------------
ts_avg_pre:   82.091 ms
ts_avg_infer: 281.546 ms
ts_avg_post:  53.842 ms
================================
Total time:   12808.000 ms
```

⚪ ppocr_mb_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-10-31 18:14:22 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
chocr_rec Build at 2024-10-31 18:05:38 For platform cv180x
Max SharedMem size:908000
find shared memory(8179200),  saved:908000 
ts_model_load: 772.000 ms
>> progress: ...................
ts_model_unload: 92.000 ms
================================
n_img:        19
n_crop:       41
--------------------------------
ts_img_load:  3833.770 ms
ts_img_crop:  314.424 ms
ts_det_pre:   1567.648 ms
ts_det_infer: 4259.731 ms
ts_det_post:  428.315 ms
ts_rec_pre:   64.891 ms
ts_rec_infer: 934.403 ms
ts_rec_post:  579.532 ms
--------------------------------
ts_avg_pre:   85.923 ms
ts_avg_infer: 273.375 ms
ts_avg_post:  53.045 ms
================================
Total time:   12936.000 ms
```
