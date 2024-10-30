# Chinese-OCR-Lite rec-model with fuse proprocess and quant to bf16

⚠ chocr 不支持竖排文本识别！

⚪ support models (BF16)

```
input.dtype:  uint8
input.shape:  (1, 32, 320, 3)   // nhwc, BGR
output.dtype: fp32              // logits
output.shape: (160, 1, 5531, 1) // lxdx
```

| model | runnable? | quality |
| :-: | :-: | :-: |
| chocr_rec_bf16 | √ | better than ppocr_det_v2 |

⚪ build & run

```shell
# compile runtime
bash ./compile_sample_runner.sh chocr_rec
# upload cvimodel & runtime
scp ./cvimodels/chocr_rec*.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_chocr_rec ../cvimodels/chocr_rec_bf16.cvimodel ./data/crop_9.jpg   ./data/chocr_keys.txt
./bin/cvi_sample_chocr_rec ../cvimodels/chocr_rec_bf16.cvimodel ./data/crop_177.jpg ./data/chocr_keys.txt

# run on chip: decode output with python
python -c "print(b''.decode())"
python -c "print(b'\xe5\xbc\x80\xe4\xbb\x99\xe5\x9b\xbd\xe6\x84\x8f\xe5\xa9\xb4\xe5\xb9\xbc\xe5\x84\xbf\xe7\x94\x9f\xe6\xb4\xbb\xe9\xa5\xb5'.decode())"
python -c "print(b'\xe5\x9b\x9e\xe7\x96\x97\xe5\xbf\xab\xe4\xbf\xae\xe9\x94\x81'.decode())"
```

⚪ chocr_rec

```
[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_chocr_rec ../cvimodels/chocr_rec_bf16.cvimodel ./data/crop_9.jpg   ./data/chocr_keys.txt
load model: 84269 clock
imgs.shape: w=238, h=30
load image: 11494 clock
load word dict: 15354 clock
preprocess: 7763 clock
feed input: 84 clock
model forward: 363 clock
------
182 3405 178 784 1436 4791 513 687 473 4682 
\xe5\xbc\x80\xe4\xbb\x99\xe5\x9b\xbd\xe6\x84\x8f\xe5\xa9\xb4\xe5\xb9\xbc\xe5\x84\xbf\xe7\x94\x9f\xe6\xb4\xbb\xe9\xa5\xb5  // "开仙国意婴幼儿生活饵"
------
postprocess: 29299 clock
unload model: 4321 clock
Total time cost: 171719
CLOCKS_PER_SEC: 1000000

[root@milkv-duo]~/tpu-sdk-cv180x-ocr/samples# ./bin/cvi_sample_chocr_rec ../cvimodels/chocr_rec_bf16.cvimodel ./data/crop_177.jpg ./data/chocr_keys.txt
load model: 89260 clock
imgs.shape: w=255, h=57
load image: 14374 clock
load word dict: 17041 clock
preprocess: 8628 clock
feed input: 80 clock
model forward: 1116 clock
------
386 2 1789 326 3265 
\xe5\x9b\x9e\xe7\x96\x97\xe5\xbf\xab\xe4\xbf\xae\xe9\x94\x81    // "回疗快修锁"
------
postprocess: 29321 clock
unload model: 4411 clock
Total time cost: 181762
CLOCKS_PER_SEC: 1000000
```
