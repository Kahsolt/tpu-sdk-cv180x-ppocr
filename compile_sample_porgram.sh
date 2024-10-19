#!/bin/bash

# 语法糖脚本：编译 samples/<prog_name> 项目

NAME=$1

if [ -z $NAME ]; then
  echo "Usage: $0 <prog_name>"
  exit -1
fi


export PATH=$PATH:/workspace/duo-sdk/riscv64-linux-musl-x86_64/bin/
TPU_SDK_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd $TPU_SDK_PATH
source ./envs_tpu_sdk.sh


echo ">> building program: $NAME"
pushd samples/$NAME
rm -rf build
mkdir -p build ; cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../../cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DTPU_SDK_PATH=$TPU_SDK_PATH \
  -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
  -DCMAKE_INSTALL_PREFIX=$TPU_SDK_PATH/samples
make
file cvi_sample_$NAME
scp cvi_sample_$NAME root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/bin
make install
file $TPU_SDK_PATH/samples/bin/cvi_sample_$NAME
popd
