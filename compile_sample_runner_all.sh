#!/bin/env bash

# 一键编译所有ppocr运行时！

bash ./compile_sample_runner.sh ppocr_det
bash ./compile_sample_runner.sh ppocr_rec
bash ./compile_sample_runner.sh ppocr_cls
bash ./compile_sample_runner.sh ppocr_sys

bash ./compile_sample_runner.sh chocr_rec

bash ./compile_sample_runner.sh myocr_sys
bash ./compile_sample_runner.sh myocr_sys_many
