# PPOCR cls-model with fuse proprocess and quant to int8

```shell
# compile runtime
bash ./compile_sample_porgram.sh ppocr_cls
# upload cvimodel & runtime
scp ./cvimodels/ppocr_mb_cls.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_1.jpg
./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_1_rev.jpg
./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9.jpg
./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9_rev.jpg
```

⚪ ppocr_mb_cls

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_1.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-19 21:20:26 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 14.940 ms)
load image (time cost: 16.042 ms)
preprocess (time cost: 11.618 ms)
feed input (time cost: 0.218 ms)
CVI_NN_Forward succeeded (time cost: 0.364 ms)
------
  0.000000, idx 0, 0
  -0.000000, idx 1, 180
------
Post-process probabilities (time cost: 0.531 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.845 ms)
Total time cost: 52.978 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_1_rev.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-19 21:20:26 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 12.698 ms)
load image (time cost: 15.844 ms)
preprocess (time cost: 8.133 ms)
feed input (time cost: 0.353 ms)
CVI_NN_Forward succeeded (time cost: 0.331 ms)
------
  0.005371, idx 0, 0
  -0.000000, idx 1, 180
------
Post-process probabilities (time cost: 0.584 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.090 ms)
Total time cost: 46.145 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-19 21:20:26 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 11.569 ms)
load image (time cost: 3.681 ms)
preprocess (time cost: 6.111 ms)
feed input (time cost: 0.319 ms)
CVI_NN_Forward succeeded (time cost: 0.337 ms)
------
  0.000000, idx 0, 0
  -0.000000, idx 1, 180
------
Post-process probabilities (time cost: 0.473 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.305 ms)
Total time cost: 29.872 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9_rev.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-19 21:20:26 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 11.431 ms)
load image (time cost: 5.088 ms)
preprocess (time cost: 6.984 ms)
feed input (time cost: 0.360 ms)
CVI_NN_Forward succeeded (time cost: 0.245 ms)
------
  0.007324, idx 0, 0
  -0.000000, idx 1, 180
------
Post-process probabilities (time cost: 0.829 ms)
CVI_NN_CleanupModel succeeded (time cost: 2.859 ms)
Total time cost: 32.126 ms
```
