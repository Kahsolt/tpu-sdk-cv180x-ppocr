# PPOCR cls-model with fuse proprocess and quant to int8

⚪ model I/O

```
input.dtype:  uint8
input.shape:  (1, 48, 640, 3)   // nhwc
output.dtype: int8              // quant-logits
output.shape: (1, 2, 1, 1)      // ndxx
```

⚪ build & run

```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_cls
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_cls.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

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
ppocr_mb_cls Build at 2024-10-20 16:07:13 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 12.946 ms)
load image (time cost: 11.483 ms)
preprocess (time cost: 6.528 ms)
feed input (time cost: 0.347 ms)
CVI_NN_Forward succeeded (time cost: 0.263 ms)
------
  q-logit for label 0: 19
  q-logit for label 180: -19
------
Show q-logits (time cost: 0.244 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.404 ms)
Total time cost: 39.496 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_1_rev.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-20 16:07:13 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 10.469 ms)

load image (time cost: 13.583 ms)
preprocess (time cost: 6.514 ms)
feed input (time cost: 0.347 ms)
CVI_NN_Forward succeeded (time cost: 0.396 ms)
------
  q-logit for label 0: -29
  q-logit for label 180: 28
------
Show q-logits (time cost: 0.245 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.032 ms)
Total time cost: 38.649 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-20 16:07:13 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 13.117 ms)
load image (time cost: 3.254 ms)
preprocess (time cost: 4.655 ms)
feed input (time cost: 0.392 ms)
CVI_NN_Forward succeeded (time cost: 0.169 ms)
------
  q-logit for label 0: 31
  q-logit for label 180: -31
------
Show q-logits (time cost: 0.240 ms)
CVI_NN_CleanupModel succeeded (time cost: 3.240 ms)
Total time cost: 29.202 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_cls ../cvimodels/ppocr_mb_cls.cvimodel ./data/crop_9_rev.jpg
version: 1.4.0
ppocr_mb_cls Build at 2024-10-20 16:07:13 For platform cv180x
Max SharedMem size:296960
CVI_NN_RegisterModel succeeded (time cost: 10.655 ms)
load image (time cost: 3.885 ms)
preprocess (time cost: 5.331 ms)
feed input (time cost: 0.344 ms)
CVI_NN_Forward succeeded (time cost: 0.178 ms)
------
  q-logit for label 0: -32
  q-logit for label 180: 31
------
Show q-logits (time cost: 0.071 ms)
CVI_NN_CleanupModel succeeded (time cost: 2.971 ms)
Total time cost: 27.080 ms
```
