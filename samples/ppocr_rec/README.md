# PPOCR rec-model with fuse proprocess and quant to bf16

```shell
# compile runtime
bash ./compile_sample_porgram.sh ppocr_rec
# upload cvimodel & runtime
scp ./cvimodels/ppocrv2_rec.cvimodel  root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels
scp ./cvimodels/ppocr_mb_rec.cvimodel root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/cvimodels

# run on chip
cd samples
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocrv2_rec.cvimodel  ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_1.jpg ./data/ppocr_keys_v1.txt
./bin/cvi_sample_ppocr_rec ../cvimodels/ppocr_mb_rec.cvimodel ./data/crop_9.jpg ./data/ppocr_keys_v1.txt
```

⚪ ppocrv2_rec

⚪ ppocr_mb_rec
