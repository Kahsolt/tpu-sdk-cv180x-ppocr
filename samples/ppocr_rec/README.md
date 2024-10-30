# PPOCR rec-model with fuse proprocess and quant to bf16

⚪ support models (BF16)

```
input.dtype:  uint8
input.shape:  (1, 48, 640, 3)   // nhwc, BGR
output.dtype: fp32              // logits
output.shape: (1, 160, 6625, 1) // nldx
```

| model | runnable? | quality |
| :-: | :-: | :-: |
| ppocr_mb_rec_bf16 | √ | just so so |

⚪ build & run

```shell
# compile runtime
bash ./compile_sample_runner.sh ppocr_rec
# upload cvimodel & runtime
scp ./cvimodels/ppocr*_rec*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_9.jpg ./data/ppocr_keys_v1.txt

# run on chip: decode output with python
python -c "print(b''.decode())"
python -c "print(b'\x30\x38\x30\x38\x31\x38\x32\x2e\x30'.decode())"
python -c "print(b'\x4d\x79\xe8\xba\xab\xe8\x88\xaa\xe4\xbb\xb7\xe5\x84\xbf\xe5\x94\xae\xe6\xb4\xbb\xe9\xa6\x86'.decode())"
```

⚪ ppocr_mb_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-29 17:12:26 For platform cv180x
Max SharedMem size:2150720
load model: 164621 clock
imgs.shape: w=618, h=67
load image: 17413 clock
load word dict: 20196 clock
preprocess: 10752 clock
feed input: 333 clock
model forward: 670 clock
------
26 27 26 27 93 27 25 466 26 
\x30\x38\x30\x38\x31\x38\x32\x2e\x30    // "0808182.0"
------
postprocess: 68411 clock
unload model: 4403 clock
Total time cost: 295123
CLOCKS_PER_SEC: 1000000
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
version: 1.4.0
ppocr_mb_rec Build at 2024-10-29 17:12:26 For platform cv180x
Max SharedMem size:2150720
load model: 336037 clock
imgs.shape: w=238, h=30
load image: 11448 clock
load word dict: 20605 clock
preprocess: 8661 clock
feed input: 246 clock
model forward: 551 clock
------
3639 4546 1350 87 1029 513 2587 473 2504 
\x4d\x79\xe8\xba\xab\xe8\x88\xaa\xe4\xbb\xb7\xe5\x84\xbf\xe5\x94\xae\xe6\xb4\xbb\xe9\xa6\x86    // "My身航价儿售活馆"
------
postprocess: 69440 clock
unload model: 4622 clock
Total time cost: 459351
CLOCKS_PER_SEC: 1000000
```
