#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>
#include <cnpy.h>

// https://github.com/sophgo/cviruntime/blob/master/doc/cvitek_tpu_sdk_development_manual.md
// https://github.com/sophgo/cviruntime/blob/master/samples/samples_extra/detector_yolov5_fused_preprocess/detector_yolov5_fused_preprocess.cpp

#define clock_to_ms(x) (x / CLOCKS_PER_SEC * 1000)
#define IMG_RESIZE_DIMS 640

int main(int argc, char **argv) {
  clock_t ts_init = clock(), ts_start;
  double ts_timecost;

  if (argc != 3) {
    printf("Usage:\n");
    printf("   %s cvimodel image.jpg\n", argv[0]);
    exit(-1);
  }

  // load model file
  const char *model_file = argv[1];
  CVI_MODEL_HANDLE model = nullptr;
  ts_start = clock();
  int ret = CVI_NN_RegisterModel(model_file, &model);
  ts_timecost = clock() - ts_start;
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    exit(1);
  }
  printf("CVI_NN_RegisterModel succeeded (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // get input output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);
  CVI_TENSOR *input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors,  input_num ); assert(input);
  CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num); assert(output);
  //printf("input.dtype: %d\n",   input->fmt);        // CVI_FMT_UINT8=7
  //printf("input.device: %d\n",  input->mem_type);   // CVI_MEM_SYSTEM=1
  //printf("output.dtype: %d\n",  output->fmt);       // CVI_FMT_FP32=0
  //printf("output.device: %d\n", output->mem_type);  // CVI_MEM_SYSTEM=1

  // load image file
  ts_start = clock();
  cv::Mat image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image: %s\n", argv[2]);
    exit(1);
  }
  ts_timecost = clock() - ts_start;
  printf("load image (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // resize + Packed2Planar
  ts_start = clock();
  cv::resize(image, image, cv::Size(IMG_RESIZE_DIMS, IMG_RESIZE_DIMS)); // linear is default
  cv::Mat channels[3];
  for (int i = 0; i < 3; i++) {
    channels[i] = cv::Mat(image.rows, image.cols, CV_8SC1);   // 8bit signed single channel
  }
  cv::split(image, channels);
  ts_timecost = clock() - ts_start;
  printf("preprocess (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // write to input tensor (int8, since we set `model_deploy.py --quant_input`)
  ts_start = clock();
  CVI_SHAPE shape = CVI_NN_TensorShape(input);  // bcwh
  size_t height = shape.dim[2], width = shape.dim[3];
  int8_t *ptr = (int8_t *) CVI_NN_TensorPtr(input);
  size_t image_size = height * width;
  for (int i = 0; i < 3; ++i) {
    memcpy(ptr + i * image_size, channels[i].data, image_size);
  }
  ts_timecost = clock() - ts_start;
  printf("feed input (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // run model inference
  ts_start = clock();
  CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  ts_timecost = clock() - ts_start;
  printf("CVI_NN_Forward succeeded (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // read from output tensor (float32, but why?)
  ts_start = clock();
  const double *results = (double *) CVI_NN_TensorPtr(output);
  cnpy::npy_save("output.npy", results, {height, width}, "w");
  ts_timecost = clock() - ts_start;
  printf("Save results to file output.npz (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // unload model
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  ts_timecost = clock() - ts_start;
  printf("CVI_NN_CleanupModel succeeded (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  ts_timecost = double(clock() - ts_init);
  printf("Total time cost: %.3f ms\n", clock_to_ms(ts_timecost));

  return 0;
}
