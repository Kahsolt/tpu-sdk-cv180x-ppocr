# PPOCR rec-model with fuse proprocess and quant to bf16

⚪ model I/O

```
input.dtype:  uint8
input.shape:  (1, 48, 640, 3)   // nhwc
output.dtype: fp32              // logits
output.shape: (1, 160, 6625, 1) // nldx
```

⚪ build & run

```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_rec
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_rec.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_9.jpg ./data/ppocr_keys_v1.txt

# decode output with python
python -c "print(b''.decode())"
python -c "print(b'\xe6\xbb\x94\x6e\xe5\xb2\x81\xe6\x99\x8f\xe7\x9f\xbf\xe6\xa4\x8b\xe7\xac\xbc\xe8\xb0\x83\xe6\x8b\xad'.decode())"
```

⚪ ppocrv2_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocrv2_rec Build at 2024-10-20 16:29:32 For platform cv180x
Max SharedMem size:2949120
CVI_NN_RegisterModel succeeded (time cost: 1154.237 ms)
load image (time cost: 19.375 ms)
preprocess (time cost: 10.452 ms)
feed input (time cost: 0.334 ms)
CVI_NN_Forward succeeded (time cost: 0.473 ms)
load word dict succeeded (time cost: 20.872 ms)
------


------
Post-process probabilities (time cost: 75.785 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.498 ms)
Total time cost: 1294.862 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocrv2_rec Build at 2024-10-20 16:29:32 For platform cv180x
Max SharedMem size:2949120
CVI_NN_RegisterModel succeeded (time cost: 1205.083 ms)
load image (time cost: 10.905 ms)
preprocess (time cost: 9.124 ms)
feed input (time cost: 0.340 ms)
CVI_NN_Forward succeeded (time cost: 0.540 ms)
load word dict succeeded (time cost: 20.929 ms)
------
93 26 
\x31\x30		// "10"
------
Post-process probabilities (time cost: 76.095 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.789 ms)
Total time cost: 1336.028 ms
```

⚪ ppocr_mb_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-20 16:06:29 For platform cv180x
Max SharedMem size:2150720
CVI_NN_RegisterModel succeeded (time cost: 164.964 ms)
load image (time cost: 18.735 ms)
preprocess (time cost: 10.848 ms)
feed input (time cost: 0.343 ms)
CVI_NN_Forward succeeded (time cost: 0.663 ms)
load word dict succeeded (time cost: 20.243 ms)
------
26 27 26 27 93 27 25 466 26 
\x38\x30\x38\x31\x38\x32\x2e\x30		// "808182.0"
------
Post-process probabilities (time cost: 77.074 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.673 ms)
Total time cost: 305.605 ms

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-20 16:06:29 For platform cv180x
Max SharedMem size:2150720
CVI_NN_RegisterModel succeeded (time cost: 156.758 ms)
load image (time cost: 10.793 ms)
preprocess (time cost: 9.031 ms)
feed input (time cost: 0.309 ms)
CVI_NN_Forward succeeded (time cost: 0.654 ms)
load word dict succeeded (time cost: 28.437 ms)
------
3639 4546 1350 87 1029 513 2587 473 2504 
\x4d\x79\xe8\xba\xab\xe8\x88\xaa\xe4\xbb\xb7\xe5\x84\xbf\xe5\x94\xae\xe6\xb4\xbb\xe9\xa6\x86		// "My身航价儿售活馆"
------
Post-process probabilities (time cost: 76.324 ms)
CVI_NN_CleanupModel succeeded (time cost: 4.267 ms)
Total time cost: 294.716 ms
```
