# PPOCR det-model with fuse proprocess and quant to int8

⚪ model I/O

```
input.dtype:  uint8
input.shape:  (1, 640, 640, 3)    // nhwc
output.dtype: int8                // quant-logits
output.shape: (1, 1, 640, 640)    // nchw
```

⚪ build & run
```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_det
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_det.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel  ./data/gt_7148.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel  ./data/gt_7148.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel  ./data/gt_7148.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/gt_7148.jpg
```

⚪ ppocrv4_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel  ./data/gt_97.jpg
imodels/ppocr_mb_det.cvimodel ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/gt_7148.jpgversion: 1.4.0
ppocrv4_det Build at 2024-10-20 15:36:29 For platform cv180x
Max SharedMem size:13107200
CVI_NN_RegisterModel succeeded (time cost: 310.776 ms)
load image (time cost: 164.670 ms)
preprocess (time cost: 91.759 ms)
feed input (time cost: 2.636 ms)
CVI_NN_Forward succeeded (time cost: 0.696 ms)
Save q-logits to file output.npz (time cost: 3.032 ms)
CVI_NN_CleanupModel succeeded (time cost: 7.039 ms)
Total time cost: 593.946 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel  ./data/gt_7148.jpg
version: 1.4.0
ppocrv4_det Build at 2024-10-20 15:36:29 For platform cv180x
Max SharedMem size:13107200
CVI_NN_RegisterModel succeeded (time cost: 201.360 ms)
load image (time cost: 203.154 ms)
preprocess (time cost: 88.560 ms)
feed input (time cost: 2.371 ms)
CVI_NN_Forward succeeded (time cost: 1.415 ms)
Save q-logits to file output.npz (time cost: 4.841 ms)
CVI_NN_CleanupModel succeeded (time cost: 6.916 ms)
Total time cost: 521.314 ms
```

⚪ ppocrv3_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel  ./data/gt_97.jpg
version: 1.4.0
ppocrv3_det Build at 2024-10-20 15:50:20 For platform cv180x
Max SharedMem size:8793600
CVI_NN_RegisterModel succeeded (time cost: 91.574 ms)
load image (time cost: 172.608 ms)
preprocess (time cost: 94.294 ms)
feed input (time cost: 2.375 ms)
CVI_NN_Forward succeeded (time cost: 2.021 ms)
Save q-logits to file output.npz (time cost: 3.985 ms)
CVI_NN_CleanupModel succeeded (time cost: 6.129 ms)
Total time cost: 382.075 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel  ./data/gt_7148.jpg
version: 1.4.0
ppocrv3_det Build at 2024-10-20 15:50:20 For platform cv180x
Max SharedMem size:8793600
CVI_NN_RegisterModel succeeded (time cost: 95.858 ms)
load image (time cost: 203.913 ms)
preprocess (time cost: 91.440 ms)
feed input (time cost: 3.127 ms)
CVI_NN_Forward succeeded (time cost: 0.878 ms)
Save q-logits to file output.npz (time cost: 3.566 ms)
CVI_NN_CleanupModel succeeded (time cost: 6.478 ms)
Total time cost: 414.324 ms
```

⚪ ppocrv2_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel  ./data/gt_97.jpg
version: 1.4.0
ppocrv2_det Build at 2024-10-20 15:55:59 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 92.569 ms)
load image (time cost: 169.981 ms)
preprocess (time cost: 93.206 ms)
feed input (time cost: 2.660 ms)
CVI_NN_Forward succeeded (time cost: 0.508 ms)
Save q-logits to file output.npz (time cost: 2.500 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.122 ms)
Total time cost: 372.522 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel  ./data/gt_7148.jpg
version: 1.4.0
ppocrv2_det Build at 2024-10-20 15:55:59 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 90.864 ms)
load image (time cost: 205.655 ms)
preprocess (time cost: 89.927 ms)
feed input (time cost: 2.724 ms)
CVI_NN_Forward succeeded (time cost: 0.468 ms)
Save q-logits to file output.npz (time cost: 3.382 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.103 ms)
Total time cost: 403.641 ms
```

⚪ ppocr_mb_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/gt_97.jpg
version: 1.4.0
ppocr_mb_det Build at 2024-10-20 16:04:43 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 89.151 ms)
load image (time cost: 169.012 ms)
preprocess (time cost: 92.562 ms)
feed input (time cost: 2.655 ms)
CVI_NN_Forward succeeded (time cost: 0.516 ms)
Save q-logits to file output.npz (time cost: 2.490 ms)
CVI_NN_CleanupModel succeeded (time cost: 5.539 ms)
Total time cost: 369.415 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/gt_7148.jpg
version: 1.4.0
ppocr_mb_det Build at 2024-10-20 16:04:43 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 91.373 ms)
load image (time cost: 203.300 ms)
preprocess (time cost: 90.249 ms)
feed input (time cost: 2.641 ms)
CVI_NN_Forward succeeded (time cost: 0.490 ms)
Save q-logits to file output.npz (time cost: 3.686 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.125 ms)
Total time cost: 402.688 ms
```
