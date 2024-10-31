# PPOCR rec-model with fuse proprocess and quant to bf16

⚪ support models (BF16)

```
input.dtype:  uint8
input.shape:  (1, 32, 320, 3)   // nhwc, BGR
output.dtype: fp32              // logits
output.shape: (1, 80, 6625, 1) // nldx
```

| model | runnable? | quality |
| :-: | :-: | :-: |
| ppocr_mb_rec_bf16 | √ | good! |

⚪ build & run

```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_rec
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_rec*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_9.jpg   ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_177.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_1.jpg   ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_58.jpg  ./data/ppocr_keys_v1.txt

# run on chip: decode output with python
python -c "print(b''.decode())"
python -c "print(b'\xe5\xbc\x80\xe5\xbf\x83\xe6\x9c\x80\xe6\x84\x8f\xe5\xa9\xb4\xe5\xb9\xbc\xe5\x84\xbf\xe7\x94\x9f\xe6\xb4\xbb\xe9\xa6\x86'.decode())"               # 开心最意婴幼儿生活馆
python -c "print(b'\xe5\x8e\x9f\xe5\xae\xb6\xe7\xae\xa1\xe9\xa5\xae\xe8\xbf\x9e\xe9\x94\x81'.decode())"                                                               # 原家管饮连锁
python -c "print(b'\xe6\x95\xac\xe6\x88\xbf\xe9\x97\xb4\xe6\x83\x85\xe8\xb5\x84\xe5\xa5\x87\xe8\xae\xb0\xe6\x9c\x8d\xe5\x8a\xa1\xe4\xb8\xad\xe5\xbf\x83'.decode())"   # 敬房间情资奇记服务中心
python -c "print(b'\xe5\x85\x89\xe8\xbe\x89\xe9\x87\x8c\xe6\xb7\xae\xe7\x89\xa9\xe4\xb8\x9a\xe5\xb0\x8f\xe5\x8c\xba'.decode())"                                       # 光辉里淮物业小区
```

⚪ ppocr_mb_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_9.jpg   ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
load model: 74850 clock
imgs.shape: w=238, h=30
load image: 7833 clock
load word dict: 19097 clock
preprocess: 4868 clock
feed input: 87 clock
model forward: 584 clock
------
182 78 1070 784 1435 4790 513 687 473 2504 
\xe5\xbc\x80\xe5\xbf\x83\xe6\x9c\x80\xe6\x84\x8f\xe5\xa9\xb4\xe5\xb9\xbc\xe5\x84\xbf\xe7\x94\x9f\xe6\xb4\xbb\xe9\xa6\x86
------
postprocess: 35165 clock
unload model: 4281 clock
Total time cost: 153627
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_177.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
load model: 76737 clock
imgs.shape: w=255, h=57
load image: 12194 clock
load word dict: 18726 clock
preprocess: 7152 clock
feed input: 88 clock
model forward: 510 clock
------
23 1516 541 4430 2201 3264 
\xe5\x8e\x9f\xe5\xae\xb6\xe7\xae\xa1\xe9\xa5\xae\xe8\xbf\x9e\xe9\x94\x81
------
postprocess: 34658 clock
unload model: 4340 clock
Total time cost: 161734
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_1.jpg   ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
load model: 76798 clock
imgs.shape: w=618, h=67
load image: 18767 clock
load word dict: 16793 clock
preprocess: 9398 clock
feed input: 185 clock
model forward: 518 clock
------
5243 22 201 521 51 799 707 310 525 194 78 
\xe6\x95\xac\xe6\x88\xbf\xe9\x97\xb4\xe6\x83\x85\xe8\xb5\x84\xe5\xa5\x87\xe8\xae\xb0\xe6\x9c\x8d\xe5\x8a\xa1\xe4\xb8\xad\xe5\xbf\x83
------
postprocess: 34917 clock
unload model: 4382 clock
Total time cost: 168866
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_58.jpg  ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-31 19:56:20 For platform cv180x
Max SharedMem size:1075360
load model: 80842 clock
imgs.shape: w=260, h=42
load image: 10516 clock
load word dict: 17113 clock
preprocess: 6364 clock
feed input: 87 clock
model forward: 375 clock
------
722 3296 86 3586 1449 85 313 492 
\xe5\x85\x89\xe8\xbe\x89\xe9\x87\x8c\xe6\xb7\xae\xe7\x89\xa9\xe4\xb8\x9a\xe5\xb0\x8f\xe5\x8c\xba
------
postprocess: 34778 clock
unload model: 4274 clock
Total time cost: 161840
CLOCKS_PER_SEC: 1000000
```
