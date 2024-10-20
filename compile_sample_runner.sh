#!/bin/env bash

# 语法糖脚本：编译 samples/<runner_name> 项目

NAME=$1

if [ -z $NAME ]; then
  echo "Usage: ./$0 <runner_name>"
  exit -1
fi
RUNNER_FILE=cvi_sample_$NAME


TPU_SDK_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
source ./envs_tpu_sdk.sh

# get cross-build toolchain
[ ! -d duo-sdk ] && git clone https://github.com/milkv-duo/duo-sdk
export PATH=$PATH:$TPU_SDK_PATH/duo-sdk/riscv64-linux-musl-x86_64/bin/

echo ">> Building runner: $RUNNER_FILE"
pushd samples/$NAME > /dev/null
#rm -rf build
mkdir -p build ; cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$TPU_SDK_PATH/cmake/toolchain-riscv64-linux-musl-x86_64.cmake \
  -DTPU_SDK_PATH=$TPU_SDK_PATH \
  -DOPENCV_PATH=$TPU_SDK_PATH/opencv \
  -DCMAKE_INSTALL_PREFIX=$TPU_SDK_PATH/samples
make
echo ">> Building runner done!"

echo ">> Cache runner $RUNNER_FILE in this repo"
file $RUNNER_FILE
make install

echo ">> Upload runner $RUNNER_FILE to MilkV-Duo!"
scp $RUNNER_FILE root@192.168.42.1:/root/tpu-sdk-cv180x-ocr/samples/bin
popd > /dev/null

echo ">> All done!"
