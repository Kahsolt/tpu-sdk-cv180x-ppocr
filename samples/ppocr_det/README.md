# PPOCR det-model with fuse proprocess and quant to int8

⚪ support models (INT8)

```
input.dtype:  uint8
input.shape:  (1, 640, 640, 3)    // nhwc, BGR
output.dtype: int8                // quant-logits
output.shape: (1, 1, 640, 640)    // nchw
```

| model | runnable? | quality |
| :-: | :-: | :-: |
| ppocrv4_det_int8  | √ | very good, clear & correct |
| ppocrv3_det_int8  | √ | good, has small fragments |
| ppocrv2_det_int8  | √ | very bad, no box found |
| ppocr_mb_det_int8 | √ | bad, broken areas |

⚪ build & run
```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_det
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_det.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det_int8.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det_int8.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det_int8.cvimodel  ./data/gt_97.jpg
./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det_int8.cvimodel ./data/gt_97.jpg

# run on host: see annotated outputs
scp root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/det*.png .
```

⚪ ppocrv4_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv4_det_int8.cvimodel  ./data/gt_97.jpg
version: 1.4.0
ppocrv4_det Build at 2024-10-29 17:27:39 For platform cv180x
Max SharedMem size:13107200
load model: 289724 clock
load image: 174563 clock
preprocess: 77156 clock
feed input: 3312 clock
model forward: 2857 clock
pull output: 8 clock
found raw n_box: 6
bbox size: w=1 h=1
bbox size: w=1 h=1
bbox size: w=80 h=12
[174, 301;
 171, 265;
 274, 247;
 277, 282]
bbox size: w=87 h=15
[-2, 288;
 -5, 243;
 110, 224;
 113, 269]
bbox size: w=2 h=4
bbox size: w=122 h=14
[155, 243;
 150, 200;
 295, 159;
 300, 202]
>> save det box results to: det-box.png 
CVI_NN_CleanupModel succeeded: 19208 clock
Total time cost: 772808
CLOCKS_PER_SEC: 1000000
```

⚪ ppocrv3_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv3_det_int8.cvimodel  ./data/gt_97.jpg
version: 1.4.0
ppocrv3_det Build at 2024-10-29 17:28:15 For platform cv180x
Max SharedMem size:8793600
load model: 93407 clock
load image: 163144 clock
preprocess: 73570 clock
feed input: 2668 clock
model forward: 435 clock
pull output: 9 clock
found raw n_box: 11
bbox size: w=7 h=5
[80, 322;
 80, 311;
 93, 311;
 93, 322]
bbox size: w=7 h=1
bbox size: w=3 h=2
bbox size: w=2 h=3
bbox size: w=19 h=6
[285, 314;
 285, 298;
 314, 298;
 314, 314]
bbox size: w=105 h=12
[151, 308;
 148, 269;
 277, 246;
 280, 284]
bbox size: w=88 h=16
[-4, 290;
 -8, 242;
 110, 222;
 114, 271]
bbox size: w=126 h=15
[152, 245;
 147, 198;
 299, 157;
 304, 204]
>> save det box results to: det-box.png 
CVI_NN_CleanupModel succeeded: 14655 clock
Total time cost: 549106
CLOCKS_PER_SEC: 1000000
```

⚪ ppocrv2_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocrv2_det_int8.cvimodel  ./data/gt_97.jpg
version: 1.4.0
ppocrv2_det Build at 2024-10-30 00:11:09 For platform cv180x
Max SharedMem size:8179200
load model: 94986 clock
load image: 167434 clock
preprocess: 73393 clock
feed input: 2603 clock
model forward: 579 clock
pull output: 7 clock
found raw n_box: 0
>> save det box results to: det-box.png 
CVI_NN_CleanupModel succeeded: 11928 clock
Total time cost: 542960
CLOCKS_PER_SEC: 1000000
```

⚪ ppocr_mb_det

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_det ../cvimodels/ppocr_mb_det_int8.cvimodel ./data/gt_97.jpg
version: 1.4.0
ppocr_mb_det Build at 2024-10-29 17:28:41 For platform cv180x
Max SharedMem size:8179200
load model: 91852 clock
load image: 169883 clock
preprocess: 74264 clock
feed input: 2678 clock
model forward: 347 clock
pull output: 7 clock
found raw n_box: 7
bbox size: w=1 h=1
bbox size: w=64 h=9
[175, 298;
 172, 271;
 253, 255;
 256, 283]
bbox size: w=10 h=5
[200, 212;
 200, 199;
 218, 199;
 218, 212]
bbox size: w=2 h=2
bbox size: w=19 h=5
[222, 206;
 220, 193;
 247, 187;
 248, 200]
bbox size: w=7 h=3
>> save det box results to: det-box.png 
CVI_NN_CleanupModel succeeded: 10498 clock
Total time cost: 552092
CLOCKS_PER_SEC: 1000000
```
