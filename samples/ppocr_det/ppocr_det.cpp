#include <stdio.h>
#include <string.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// ppocr det model infer
// https://github.com/sophgo/cviruntime/blob/master/doc/cvitek_tpu_sdk_development_manual.md

#pragma region def
typedef vector<Point> Contour;
typedef vector<Point> Box;

#define min(x, y) (((x) <= (y)) ? (x) : (y))
#define max(x, y) (((x) >= (y)) ? (x) : (y))
#define clip(x, a, b) min(max((x), (a)), (b))

#define DET_IMG_SIZE      640
#define DET_SEG_THRESH    0.6
#define DET_BOX_THRESH    0.3
#define DET_MAX_BOXES     100
#define DET_MIN_SIZE      5
#define DET_UNCLIP_K      3.0
#define DET_UNCLIP_T      1.414
#define DET_QSCALE        127

//#define SHOW_MODEL_INFO
//#define SAVE_BITMAP
#pragma endregion def


int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s ppocr_det.cvimodel image.jpg\n", argv[0]);
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
  printf("  input->count: %d\n", input->count);   // 1228800
  printf("  input->shape: %d (%d, %d, %d, %d)\n", input->shape.dim_size, input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]); // 4 (1, 640, 640, 3)
  printf("  input->pixel_format: %d\n", input->pixel_format);   // CVI_NN_PIXEL_BGR_PACKED=1
  printf("  output->fmt: %d\n", output->fmt);     // CVI_FMT_INT8=6
  printf("  output->count: %d\n", output->count); // 409600
  printf("  output->shape: %d (%d, %d, %d, %d)\n", output->shape.dim_size, output->shape.dim[0], output->shape.dim[1], output->shape.dim[2], output->shape.dim[3]); // 4 (1, 1, 640, 640)
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

  // preprocess (resize)
  ts_start = clock();
  int W = im.cols, H = im.rows;
  int size_max = max(W, H);
  int cvs_size = DET_IMG_SIZE;  // fix input size!
  Mat cvs = Mat::zeros(cvs_size, cvs_size, CV_8UC3);
  if (cvs_size < size_max) {
    int W_resize = W * cvs_size / size_max, 
        H_resize = H * cvs_size / size_max;
    resize(im, im, Size(W_resize, H_resize));   // inplace!
    im.copyTo(cvs(Rect(0, 0, W_resize, H_resize)));
  } else {
    im.copyTo(cvs(Rect(0, 0, W, H)));
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

  // read from output tensor (int8)
  ts_start = clock();
  int8_t *plogits = (int8_t *) CVI_NN_TensorPtr(output);
  Mat segmap = Mat(cvs_size, cvs_size, CV_8SC1, plogits);
  printf("pull output: %d clock\n", clock() - ts_start);

  // postprocess (segmap to bitmap)
  ts_start = clock();
  Mat bitmap = segmap >= int8_t(DET_SEG_THRESH * DET_QSCALE);   // CV_8UC1, vset {0, 255}

#ifdef SAVE_BITMAP
  imwrite("det-bitmap.png", bitmap);
  cout << ">> save det bitmap to: det-bitmap.png" << endl;
#endif

  Mat det_result(im);
  vector<Contour> contours;
  findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
  int n_contours = min(contours.size(), DET_MAX_BOXES);
  cout << "found raw n_box: " << n_contours << endl;
  for (int i = 0 ; i < n_contours ; i++) {
    auto contour = contours[i];
    if (contour.size() < 4) continue;
    // bouding box
    RotatedRect bbox = minAreaRect(contour);
    int w = bbox.size.width, h = bbox.size.height;
    cout << "bbox size: w=" << w << " h=" << h << endl;
    if (min(w, h) < DET_MIN_SIZE) continue;
    Point2f tmp[4]; bbox.points(tmp); // order: bottomLeft, topLeft, topRight, bottomRight
    Box box = { tmp[0], tmp[1], tmp[2], tmp[3] };   // dtype cvt
    // box unclip (this is our magic :)
    double w_h = w + h, wh = w * h;
    double t = (sqrt(w_h * w_h + 4 * (DET_UNCLIP_K - 1) * wh) - w_h) / 4 * DET_UNCLIP_T;
    int n_pts = box.size();
    for (int  i = 0 ; i < n_pts ; i++) {
      Point &p = box[i];
      Point &p_L = box[(i - 1 + n_pts) % n_pts];
      Point &p_R = box[(i + 1)         % n_pts];
      Point v_L = p_L - p; v_L /= norm(v_L);
      Point v_R = p_R - p; v_R /= norm(v_R);
      Point2f v = (v_L + v_R) * t;
      p.x -= v.x; p.y -= v.y;
    }
    cout << box << endl;
    Scalar color(255, 255, 0);
    polylines(det_result, box, true, color, 2);
  }
  printf("post_process: %d clock\n", clock() - ts_start);

  imwrite("det-box.png", det_result);
  cout << ">> save det box results to: det-box.png " << endl;

  // unload model
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  printf("unload model: %d clock\n", clock() - ts_start);

  printf("Total time cost: %d\n", clock() - ts_init);
  printf("CLOCKS_PER_SEC: %d\n", CLOCKS_PER_SEC);
}
