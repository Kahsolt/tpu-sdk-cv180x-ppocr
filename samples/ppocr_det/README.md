# PPOCR det-model with fuse proprocess and quant to int8

```shell
# compile runtime
bash ./compile_sample_porgram.sh ppocr_det
# upload cvimodel & runtime
scp ./cvimodels/ppocrv*_det.cvimodel  root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
scp ./cvimodels/ppocr_mb_det.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel ./data/cat.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel ./data/cat.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel ./data/cat.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/cat.jpg
```

⚪ ppocrv4_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det.cvimodel ./data/cat.jpg
version: 1.4.0
ppocrv4_det Build at 2024-10-18 15:40:20 For platform cv180x
Max SharedMem size:13107200
CVI_NN_RegisterModel succeeded (time cost: 339.588 ms)
load image (time cost: 51.245 ms)
preprocess (time cost: 70.185 ms)
feed input (time cost: 2.695 ms)
CVI_NN_Forward succeeded (time cost: 1.395 ms)
Save results to file output.npz (time cost: 72.745 ms)
CVI_NN_CleanupModel succeeded (time cost: 12.098 ms)
Total time cost: 563.139 ms
```

⚪ ppocrv3_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det.cvimodel ./data/cat.jpg
version: 1.4.0
ppocrv3_det Build at 2024-10-18 15:53:30 For platform cv180x
Max SharedMem size:8793600
CVI_NN_RegisterModel succeeded (time cost: 94.522 ms)
load image (time cost: 52.353 ms)
preprocess (time cost: 71.470 ms)
feed input (time cost: 2.749 ms)
CVI_NN_Forward succeeded (time cost: 0.723 ms)
Save results to file output.npz (time cost: 67.419 ms)
CVI_NN_CleanupModel succeeded (time cost: 7.474 ms)
Total time cost: 305.132 ms
```

⚪ ppocrv2_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det.cvimodel ./data/cat.jpg
version: 1.4.0
ppocrv2_det Build at 2024-10-18 16:01:49 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 93.290 ms)
load image (time cost: 50.219 ms)
preprocess (time cost: 67.693 ms)
feed input (time cost: 2.511 ms)
CVI_NN_Forward succeeded (time cost: 0.547 ms)
Save results to file output.npz (time cost: 68.752 ms)
CVI_NN_CleanupModel succeeded (time cost: 7.931 ms)
Total time cost: 298.604 ms
```

⚪ ppocr_mb_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det.cvimodel ./data/cat.jpg
version: 1.4.0
ppocr_mb_det Build at 2024-10-18 16:10:33 For platform cv180x
Max SharedMem size:8179200
CVI_NN_RegisterModel succeeded (time cost: 95.538 ms)
load image (time cost: 52.587 ms)
preprocess (time cost: 69.422 ms)
feed input (time cost: 2.726 ms)
CVI_NN_Forward succeeded (time cost: 0.775 ms)
Save results to file output.npz (time cost: 69.853 ms)
CVI_NN_CleanupModel succeeded (time cost: 9.113 ms)
Total time cost: 308.009 ms
```
