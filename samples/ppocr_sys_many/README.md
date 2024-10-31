# PP-OCR system with fuse proprocess and quant to INT8 & BF16

一次性推理整个文件夹！其余请参考 ppocr_sys 子项目 ;)

性能优化建议:

  - 使用更小的 det 模型
  - 设置更大的 `DET_MIN_SIZE`
  - 不启用 `DEBUG_DUMP_*` 

使用自定义的 cvimodel 模型：

  - 检查 `DET_IMG_SIZE`, `REC_IMG_WIDTH`, `REC_IMG_HEIGHT` 设置是否与模型编译规格一致
  - 检查 `CVI_NN_TensorPtr` 返回值类型是否与模型编译规格一致

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
ts_model_load: 768.000 ms
>> progress: ...................
ts_model_unload: 96.000 ms
================================
n_img:        19
n_crop:       40
--------------------------------
ts_img_load:  3685.632 ms
ts_img_crop:  320.168 ms
ts_det_pre:   1674.590 ms
ts_det_infer: 4603.578 ms
ts_det_post:  431.149 ms
ts_rec_pre:   63.767 ms
ts_rec_infer: 1331.415 ms
ts_rec_post:  676.561 ms
--------------------------------
ts_avg_pre:   91.492 ms
ts_avg_infer: 312.368 ms
ts_avg_post:  58.301 ms
================================
Total time:   13720.000 ms
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
ts_model_load: 760.000 ms
>> progress: ...................
ts_model_unload: 96.000 ms
================================
n_img:        19
n_crop:       50
--------------------------------
ts_img_load:  3623.653 ms
ts_img_crop:  307.992 ms
ts_det_pre:   1688.924 ms
ts_det_infer: 4217.506 ms
ts_det_post:  365.775 ms
ts_rec_pre:   76.005 ms
ts_rec_infer: 1665.869 ms
ts_rec_post:  845.795 ms
--------------------------------
ts_avg_pre:   92.891 ms
ts_avg_infer: 309.651 ms
ts_avg_post:  63.767 ms
================================
Total time:   13688.000 ms
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
ts_model_load: 696.000 ms
>> progress: ...................
ts_model_unload: 96.000 ms
================================
n_img:        19
n_crop:       41
--------------------------------
ts_img_load:  3799.978 ms
ts_img_crop:  343.423 ms
ts_det_pre:   1860.028 ms
ts_det_infer: 4256.473 ms
ts_det_post:  594.212 ms
ts_rec_pre:   70.038 ms
ts_rec_infer: 1366.016 ms
ts_rec_post:  693.169 ms
--------------------------------
ts_avg_pre:   101.582 ms
ts_avg_infer: 295.920 ms
ts_avg_post:  67.757 ms
================================
Total time:   13848.000 ms
```
