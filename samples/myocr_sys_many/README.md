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

./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv4_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0   # OOM
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0
./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/chocr_rec_bf16.cvimodel /dataset/train_full_images_0

# run on host
scp -r root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results .
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .
python convert_results.py ./results.txt
```

⚪ ppocrv3_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel 
/dataset/train_full_images_0
version: 1.4.0
ppocrv3_det Build at 2024-10-30 18:47:08 For platform cv180x
Max SharedMem size:8793600
version: 1.4.0
chocr_rec Build at 2024-10-30 19:45:05 For platform cv180x
Max SharedMem size:908000
find shared memory(8793600),  saved:908000 
ts_model_load: 916.000 ms
>> progress: ...................
ts_model_unload: 108.000 ms
================================
n_img:        19
n_crop:       40
--------------------------------
ts_img_load:  4501.404 ms
ts_img_crop:  458.962 ms
ts_det_pre:   2299.035 ms
ts_det_infer: 4738.777 ms
ts_det_post:  1110.852 ms
ts_rec_pre:   67.053 ms
ts_rec_infer: 914.731 ms
ts_rec_post:  571.862 ms
--------------------------------
ts_avg_pre:   124.531 ms
ts_avg_infer: 297.553 ms
ts_avg_post:  88.564 ms
================================
Total time:   15732.000 ms
```

⚪ ppocrv2_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel 
/dataset/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-10-30 18:42:03 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
chocr_rec Build at 2024-10-30 19:45:05 For platform cv180x
Max SharedMem size:908000
find shared memory(8179200),  saved:908000 
ts_model_load: 920.000 ms
>> progress: ...................
ts_model_unload: 100.000 ms
================================
n_img:        19
n_crop:       50
--------------------------------
ts_img_load:  3787.367 ms
ts_img_crop:  349.583 ms
ts_det_pre:   1610.138 ms
ts_det_infer: 4276.350 ms
ts_det_post:  601.726 ms
ts_rec_pre:   77.552 ms
ts_rec_infer: 1140.498 ms
ts_rec_post:  721.855 ms
--------------------------------
ts_avg_pre:   88.826 ms
ts_avg_infer: 285.097 ms
ts_avg_post:  69.662 ms
================================
Total time:   13632.000 ms
```

⚪ ppocr_mb_det + chocr_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_myocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/chocr_rec_bf16.cvimodel 
/dataset/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-10-30 18:42:34 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
chocr_rec Build at 2024-10-30 19:45:05 For platform cv180x
Max SharedMem size:908000
find shared memory(8179200),  saved:908000 
ts_model_load: 828.000 ms
>> progress: ...................
ts_model_unload: 88.000 ms
================================
n_img:        19
n_crop:       41
--------------------------------
ts_img_load:  4041.439 ms
ts_img_crop:  350.884 ms
ts_det_pre:   1937.523 ms
ts_det_infer: 4307.407 ms
ts_det_post:  810.919 ms
ts_rec_pre:   67.760 ms
ts_rec_infer: 937.045 ms
ts_rec_post:  602.651 ms
--------------------------------
ts_avg_pre:   105.541 ms
ts_avg_infer: 276.024 ms
ts_avg_post:  74.398 ms
================================
Total time:   14080.000 ms
```
