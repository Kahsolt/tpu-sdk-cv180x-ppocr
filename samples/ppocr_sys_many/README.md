# PP-OCR system with fuse proprocess and quant to INT8 & BF16

一次性推理整个文件夹！其余请参考 ppocr_sys 子项目 ;)

性能优化建议:

  - 使用更小的 det 模型
  - 设置更大的 `DET_MIN_SIZE` (查全率受损!)
  - 不启用 `DEBUG_DUMP_*` 

使用自定义的 cvimodel 模型：

  - 检查 `DET_IMG_SIZE`, `REC_IMG_WIDTH`, `REC_IMG_HEIGHT` 设置是否与模型编译规格一致
  - 检查 `CVI_NN_TensorPtr` 返回值类型是否与模型编译规格一致
  - 检查 `L` 和 `D` 的索引位置是否与模型编译规格一致

```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_sys_many
# upload cvimodel & runtime
scp ./cvimodels/ppocr*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
# upload infer data
ssh root@192.168.42.1 "mkdir -p /dataset/train_full_images_0"
scp /path/to/img/gt_100*.jpg root@192.168.42.1:/dataset/train_full_images_0   # 19 images

# run on chip
source ./envs_tpu_sdk.sh
cd samples
# run with the highest priority! :)
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv4_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0   # OOM
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0
nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0

# run on host
scp -r root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results .
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/results.txt .
python convert_results.py ./results.txt
```

⚪ ppocrv3_det + ppocr_mb_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv3_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocrv3_det Build at 2024-10-31 18:52:24 For platform cv180x
Max SharedMem size:8793600
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
find shared memory(8793600),  saved:1075360 
ts_model_load: 861.625 ms
>> progress: ...................
ts_model_unload: 98.313 ms
================================
n_img:        19
n_crop:       145
--------------------------------
ts_img_load:  3601.535 ms
ts_img_crop:  615.459 ms
ts_det_pre:   1618.865 ms
ts_det_infer: 4618.099 ms
ts_det_post:  380.009 ms
ts_rec_pre:   173.113 ms
ts_rec_infer: 4836.261 ms
ts_rec_post:  857.765 ms
--------------------------------
ts_avg_pre:   94.315 ms
ts_avg_infer: 497.598 ms
ts_avg_post:  65.146 ms
================================
Total time:   17722.326 ms
```

⚪ ppocrv2_det + ppocr_mb_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocrv2_det_int8.cvimodel  ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocrv2_det Build at 2024-10-31 18:27:55 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 1266.375 ms
>> progress: ...................
ts_model_unload: 91.480 ms
================================
n_img:        19
n_crop:       137
--------------------------------
ts_img_load:  3714.881 ms
ts_img_crop:  476.227 ms
ts_det_pre:   1849.673 ms
ts_det_infer: 4250.347 ms
ts_det_post:  433.215 ms
ts_rec_pre:   160.201 ms
ts_rec_infer: 4566.401 ms
ts_rec_post:  809.094 ms
--------------------------------
ts_avg_pre:   105.783 ms
ts_avg_infer: 464.039 ms
ts_avg_post:  65.385 ms
================================
Total time:   17684.490 ms
```

⚪ ppocr_mb_det + ppocr_mb_rec

```shell
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# nice -n -19 ./bin/cvi_sample_ppocr_sys_many ../cvimodels/ppocr_mb_det_int8.cvimodel ../cvimodels/ppocr_mb_rec_bf16.cvimodel /dataset/train_full_images_0
version: 1.4.0
ppocr_mb_det Build at 2024-10-31 18:14:22 For platform cv180x
Max SharedMem size:8179200
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
find shared memory(8179200),  saved:1075360 
ts_model_load: 727.375 ms
>> progress: ...................
ts_model_unload: 134.683 ms
================================
n_img:        19
n_crop:       95
--------------------------------
ts_img_load:  3529.004 ms
ts_img_crop:  413.481 ms
ts_det_pre:   1647.848 ms
ts_det_infer: 4244.373 ms
ts_det_post:  321.652 ms
ts_rec_pre:   117.571 ms
ts_rec_infer: 3163.786 ms
ts_rec_post:  561.299 ms
--------------------------------
ts_avg_pre:   92.917 ms
ts_avg_infer: 389.903 ms
ts_avg_post:  46.471 ms
================================
Total time:   14942.379 ms
```
