# tpu-sdk-cv180x-ppocr

    Run Paddle-OCR via cvimodel on MilkV-Duo! 

----

ℹ The prebuilt [cvimodels](./cvimodels) in the repo comes from: https://github.com/Kahsolt/CCF-BDCI-2024-TPU-ocr-deploy

To compiler a cvimodel runner, use my handy script:

```shell
# this will compile & auto upload to your chip :)
bash ./compile_sample_runner.sh myocr_sys
```

or run step by step, for example:

```shell
# compile an ocr TPU app
pushd samples/myocr_sys
mkdir -p build ; cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DTPU_SDK_PATH=$TPU_SDK_PATH \
  -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
  -DCMAKE_INSTALL_PREFIX=$TPU_SDK_PATH/samples
make
make install
file $TPU_SDK_PATH/samples/bin/cvi_sample_myocr_sys
popd

# upload the entir tpu-sdk-cv180x-ocr to chip
scp -r /workspace/tpu-sdk-cv180x-ocr root@192.168.42.1:/root

# run on chip
cd tpu-sdk-cv180x-ocr
source ./envs_tpu_sdk.sh
cd samples
./bin/cvi_sample_myocr_sys ../cvimodels/ppocrv4_det_int8.cvimodel  ../cvimodels/chocr_rec_bf16.cvimodel ./data/gt_97.jpg
```

#### references

- https://github.com/milkv-duo/tpu-sdk-cv180x
- https://github.com/sophgo/cviruntime
- cv::imread is slow 
  - https://github.com/libvips/pyvips/issues/179#issuecomment-618936358
  - https://github.com/libvips/libvips

----
by Armit
2024/10/19
