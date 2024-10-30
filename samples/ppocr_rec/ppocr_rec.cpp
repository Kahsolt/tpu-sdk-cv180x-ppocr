#include <stdio.h>
#include <time.h>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// ppocr rec model infer
// https://github.com/sophgo/cviruntime/blob/master/doc/cvitek_tpu_sdk_development_manual.md
// https://github.com/sophgo/cviruntime/blob/master/samples/classifier_fused_preprocess/classifier_fused_preprocess.cpp

#pragma region def
#define min(x, y) (((x) <= (y)) ? (x) : (y))

#define REC_IMG_HEIGHT   48
#define REC_IMG_WIDTH    640
#define REC_LOGIT_NINF   -114514191981.0

//#define SHOW_MODEL_INFO
#pragma endregion def


int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s ppocr_rec.cvimodel image.jpg ppocr_keys_v1.txt\n", argv[0]);
    exit(-1);
  }

  clock_t ts_init = clock(), ts_start;

  // load model
  CVI_MODEL_HANDLE model = nullptr;
  ts_start = clock();
  int ret = CVI_NN_RegisterModel(argv[1], &model);
  if (ret != CVI_RC_SUCCESS) {
    printf("CVI_NN_RegisterModel failed, err %d\n", ret);
    exit(1);
  }
  printf("load model: %d clock\n", clock() - ts_start);

  // get input output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);
  CVI_TENSOR *input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, input_tensors,  input_num ); assert(input);
  CVI_TENSOR *output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, output_tensors, output_num); assert(output);

#ifdef SHOW_MODEL_INFO
  printf("[Model Info]\n");
  printf("  input->fmt: %d\n", input->fmt);       // CVI_FMT_UINT8=7
  printf("  input->count: %d\n", input->count);   // 92160
  printf("  input->shape: %d (%d, %d, %d, %d)\n", input->shape.dim_size, input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]); // 4 (1, 48, 640, 3)
  printf("  input->pixel_format: %d\n", input->pixel_format);   // CVI_NN_PIXEL_BGR_PACKED=1
  printf("  output->fmt: %d\n", output->fmt);     // CVI_FMT_FP32=0
  printf("  output->count: %d\n", output->count); // 1060000
  printf("  output->shape: %d (%d, %d, %d, %d)\n", output->shape.dim_size, output->shape.dim[0], output->shape.dim[1], output->shape.dim[2], output->shape.dim[3]); // 4 (1, 160, 6625, 1)
  printf("  output->pixel_format: %d\n", output->pixel_format); // CVI_NN_PIXEL_TENSOR=100
#endif

  // load image
  ts_start = clock();
  Mat im = imread(argv[2]);
  if (!im.data) {
    printf("Could not open or find the image: %s\n", argv[2]);
    exit(2);
  }
  printf("imgs.shape: w=%d, h=%d\n", im.cols, im.rows);
  printf("load image: %d clock\n", clock() - ts_start);

  // load word dict
  ts_start = clock();
  ifstream file(argv[3]);
  if (!file) {
    printf("File not exist %s\n", argv[3]);
    exit(3);
  }
  vector<string> labels;
  string line;
  while (getline(file, line)) labels.push_back(string(line));
  printf("load word dict: %d clock\n", clock() - ts_start);

  // preprocess (resize)
  ts_start = clock();
  Mat cvs;
  if (true) {
    // just resize, since padding this input will cause wrong predict??!
    resize(im, cvs, Size(REC_IMG_WIDTH, REC_IMG_HEIGHT));
  } else {
    int W = im.cols, H = im.rows;       // fix H to 48
    int W_resize = min(W * REC_IMG_HEIGHT / H, REC_IMG_WIDTH);
    if (W != W_resize || H != REC_IMG_HEIGHT)
      resize(im, im, Size(W_resize, REC_IMG_HEIGHT));
    cvs = Mat::zeros(REC_IMG_HEIGHT, REC_IMG_WIDTH, CV_8UC3);
    im.copyTo(cvs(Rect(0, 0, W_resize, REC_IMG_HEIGHT)));
  }
  printf("preprocess: %d clock\n", clock() - ts_start);

  // write to input tensor (uint8)
  ts_start = clock();
  uint8_t *ptr = (uint8_t *) CVI_NN_TensorPtr(input);
  memcpy(ptr, cvs.reshape(1).data, cvs.step[0] * cvs.rows);
  printf("feed input: %d clock\n", clock() - ts_start);

  // run model inference
  ts_start = clock();
  CVI_NN_Forward(model, input_tensors, input_num, output_tensors, output_num);
  printf("model forward: %d clock\n", clock() - ts_start);

  // read from output tensor (float32)
  float_t *logits = (float_t *) CVI_NN_TensorPtr(output);   // [B=1, L=160, D=6625, X=1]

  // post-process: argmax()
  ts_start = clock();
  const size_t L = output->shape.dim[1], D = output->shape.dim[2];
  int last_idx = -1;
  printf("------\n");
  last_idx = -1;
  for (int k = 0 ; k < L; k++) {
    float_t *logdist = &logits[k * D];
    float_t logit_max = REC_LOGIT_NINF;
    int idx = -1;
    for (int i = 0 ; i < D; i++) {
      float_t logit = logdist[i];
      if (logit > logit_max) {
        logit_max = logit;
        idx = i;
      }
    }
    if (idx == -1) break;
    if (idx != 0 && idx != last_idx) {
      printf("%d ", idx);
      last_idx = idx;
    }
  }
  putchar('\n');
  last_idx = -1;
  for (int k = 0 ; k < L; k++) {
    float_t *logdist = &logits[k * D];
    float_t logit_max = REC_LOGIT_NINF;
    int idx = -1;
    for (int i = 0 ; i < D; i++) {
      float_t logit = logdist[i];
      if (logit > logit_max) {
        logit_max = logit;
        idx = i;
      }
    }
    if (idx == -1) break;
    if (idx != 0 && idx != last_idx) {
      // print in hex format due to chip not support cjk-characters
      string label = labels[idx - 1];  // offset by <PAD>=0
      for (int j = 0 ; j < label.size(); j++)
        printf("\\x%.2x", label[j]);
      last_idx = idx;
    }
  }
  printf("\n------\n");
  printf("postprocess: %d clock\n", clock() - ts_start);

  // unload model
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  printf("unload model: %d clock\n", clock() - ts_start);

  printf("Total time cost: %d\n", clock() - ts_init);
  printf("CLOCKS_PER_SEC: %d\n", CLOCKS_PER_SEC);
}
