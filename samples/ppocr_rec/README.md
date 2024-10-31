# PPOCR rec-model with fuse proprocess and quant to bf16

⚠ 精度很烂，会不会是 L 维度太长导致误差累计崩溃？

⚪ support models (BF16)

```
input.dtype:  uint8
input.shape:  (1, 48, 640, 3)   // nhwc, BGR
output.dtype: fp32              // logits
output.shape: (1, 160, 6625, 1) // nldx
```

| model | runnable? | quality |
| :-: | :-: | :-: |
| ppocr_mb_rec_bf16 | √ | just so so, wired input fmt??! |

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
python -c "print(b'\x4d\x79\xe8\xba\xab\xe8\x88\xaa\xe4\xbb\xb7\xe5\x84\xbf\xe5\x94\xae\xe6\xb4\xbb\xe9\xa6\x86'.decode())"   # "My身航价儿售活馆"
python -c "print(b'\x79\x6e'.decode())"                                                                                       # "yn"
python -c "print(b'\x30\x38\x30\x38\x31\x38\x32\x2e\x30'.decode())"                                                           # "0808182.0"
python -c "print(b'\xe8\x81\x98\xe7\xbb\xb4\x32\x67\xe6\x8a\x98'.decode())"                                                   # "聘维2g折"
```

⚪ ppocr_mb_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_9.jpg   ./data/ppocr_keys_v1.txt
load model: 179153 clock
imgs.shape: w=238, h=30
load image: 12375 clock
load word dict: 16313 clock
preprocess: 10335 clock
feed input: 340 clock
model forward: 615 clock
------
3639 4546 1350 87 1029 513 2587 473 2504 
\x4d\x79\xe8\xba\xab\xe8\x88\xaa\xe4\xbb\xb7\xe5\x84\xbf\xe5\x94\xae\xe6\xb4\xbb\xe9\xa6\x86
------
postprocess: 68046 clock
unload model: 4729 clock
Total time cost: 301098
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_177.jpg ./data/ppocr_keys_v1.txt
load model: 159816 clock
imgs.shape: w=255, h=57
load image: 14205 clock
load word dict: 15972 clock
preprocess: 11902 clock
feed input: 200 clock
model forward: 1401 clock
------
4546 4547 
\x79\x6e
------
postprocess: 67882 clock
unload model: 4813 clock
Total time cost: 285731
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_1.jpg   ./data/ppocr_keys_v1.txt
load model: 168782 clock
imgs.shape: w=618, h=67
load image: 20470 clock
load word dict: 16194 clock
preprocess: 11683 clock
feed input: 483 clock
model forward: 530 clock
------
26 27 26 27 93 27 25 466 26 
\x30\x38\x30\x38\x31\x38\x32\x2e\x30
------
postprocess: 68302 clock
unload model: 4576 clock
Total time cost: 300539
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec_bf16.cvimodel ./data/crop_58.jpg  ./data/ppocr_keys_v1.txt
load model: 164678 clock
imgs.shape: w=260, h=42
load image: 13189 clock
load word dict: 15534 clock
preprocess: 10532 clock
feed input: 340 clock
model forward: 526 clock
------
1709 1373 25 4548 963 
\xe8\x81\x98\xe7\xbb\xb4\x32\x67\xe6\x8a\x98
------
postprocess: 68510 clock
unload model: 4702 clock
Total time cost: 286733
CLOCKS_PER_SEC: 1000000
```
