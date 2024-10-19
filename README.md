# tpu-sdk-cv180x-ppocr

    Run Paddle-OCR via cvimodel on MilkV-Duo! 

----

```shell
# suppose we are at /workspace
pwd
# get cross-build toolchain
git clone https://github.com/milkv-duo/duo-sdk
export PATH=$PATH:/workspace/duo-sdk/riscv64-linux-musl-x86_64/bin/

# goto tpu-sdk folder
pushd tpu-sdk-cv180x-ocr
export TPU_SDK_PATH=$pwd
source ./envs_tpu_sdk.sh

# compile a hello-word TPU app
pushd samples/classifier_fused_preprocess
mkdir -p build ; cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DTPU_SDK_PATH=$TPU_SDK_PATH \
  -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
  -DCMAKE_INSTALL_PREFIX=$TPU_SDK_PATH/samples
make
make install
file $TPU_SDK_PATH/samples/bin/cvi_sample_classifier
popd

# upload the entir tpu-sdk-cv180x-ocr to chip
scp -r /workspace/tpu-sdk-cv180x-ocr root@192.168.42.1:/root

# run on chop
cd tpu-sdk-cv180x-ocr
source ./envs_tpu_sdk.sh
cd samples
./bin/cvi_sample_classifier_fused_preprocess ../cvimodels/mobilenet_v2_int8_fuse_asym.cvimodel ./data/cat.jpg ./data/synset_words.txt
```

#### references

- https://github.com/milkv-duo/tpu-sdk-cv180x
- https://github.com/sophgo/cviruntime

----
by Armit
2024/10/19
