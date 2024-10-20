#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>
#include <numeric>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

// https://github.com/sophgo/cviruntime/blob/master/doc/cvitek_tpu_sdk_development_manual.md
// https://github.com/sophgo/cviruntime/blob/master/samples/classifier_fused_preprocess/classifier_fused_preprocess.cpp

#define clock_to_ms(x) (x / CLOCKS_PER_SEC * 1000)
#define IMG_RESIZE_HEIGHT 48
#define IMG_RESIZE_WIDTH 640

int main(int argc, char **argv) {
  clock_t ts_init = clock(), ts_start;
  double ts_timecost;

  if (argc != 3) {
    printf("Usage: %s ppocr_cls.cvimodel image.jpg\n", argv[0]);
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
  //printf("input.shape: %d (%d, %d, %d, %d)\n", input->shape.dim_size, input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]);  // 4 (1, 48, 640, 3)
  //printf("output.dtype: %d\n",  output->fmt);       // CVI_FMT_INT8=6
  //printf("output.shape: %d (%d, %d, %d, %d)\n", output->shape.dim_size, output->shape.dim[0], output->shape.dim[1], output->shape.dim[2], output->shape.dim[3]);  // 4 (1, 2, 1, 1)

  // load image file
  ts_start = clock();
  cv::Mat image = cv::imread(argv[2]);
  if (!image.data) {
    printf("Could not open or find the image: %s\n", argv[2]);
    exit(2);
  }
  ts_timecost = clock() - ts_start;
  printf("load image (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // resize
  ts_start = clock();
  cv::resize(image, image, cv::Size(IMG_RESIZE_WIDTH, IMG_RESIZE_HEIGHT)); // linear is default
  ts_timecost = clock() - ts_start;
  printf("preprocess (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // write to input tensor (uint8)
  ts_start = clock();
  size_t count = CVI_NN_TensorCount(input);
  uint8_t *ptr = (uint8_t *) CVI_NN_TensorPtr(input);
  memcpy(ptr, image.reshape(1).data, count);
  ts_timecost = clock() - ts_start;
  printf("feed input (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // run model inference
  ts_start = clock();
  CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  ts_timecost = clock() - ts_start;
  printf("CVI_NN_Forward succeeded (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // read from output tensor (int8)
  ts_start = clock();
  int8_t *logits = (int8_t *) CVI_NN_TensorPtr(output);  // this model has softmax
  printf("------\n");
  const char* labels[] = {"0", "180"};
  for (size_t i = 0; i < 2; i++)
    printf("  q-logit for label %s: %d\n", labels[i], logits[i]);
  printf("------\n");
  ts_timecost = clock() - ts_start;
  printf("Show q-logits (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  // unload model
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  ts_timecost = clock() - ts_start;
  printf("CVI_NN_CleanupModel succeeded (time cost: %.3f ms)\n", clock_to_ms(ts_timecost));

  ts_timecost = double(clock() - ts_init);
  printf("Total time cost: %.3f ms\n", clock_to_ms(ts_timecost));

  return 0;
}
