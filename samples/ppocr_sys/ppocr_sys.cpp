#include <stdio.h>
#include <limits.h>
#include <time.h>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// ppocr basic pipeline: det + rec (no cls)
// https://github.com/sophgo/cviruntime/blob/master/doc/cvitek_tpu_sdk_development_manual.md
// https://docs.opencv.org/4.x/
// https://stackoverflow.com/questions/8267191/how-to-crop-a-cvmat-in-opencv

#pragma region def
#define min(x, y) ((x) <= (y)) ? (x) : (y)
#define max(x, y) ((x) >= (y)) ? (x) : (y)
#define clip(x, a, b) min(max((x), (a)), (b))

#define DET_IMG_SIZE      640
#define DET_SEG_THRESH    0.6
#define DET_BOX_THRESH    0.3
#define DET_MAX_BOXES     100
#define DET_MIN_SIZE      3
#define DET_UNCLIP_RATIO  2.0
#define DET_QSCALE        127
#define REC_IMG_WIDTH     640
#define REC_IMG_HEIGHT    48
#define REC_LOGIT_NINF    -114514191981.0

#define DEBUG_MODEL       false
#define DEBUG_OUT         false
#pragma endregion def

bool point_cmp(Point &a, Point &b) {
  return a.x != b.x ? a.x < b.x : a.y < b.y;
}
inline bool box_bound(vector<Point> &contour) {
  // minimal bouding rotatary box
  RotatedRect bbox = minAreaRect(contour);
  //cout << "raw box size: " << bbox.size.width << " " << bbox.size.height << endl;
  int msize = min(round(bbox.size.width), round(bbox.size.height));
  if (msize < DET_MIN_SIZE) return true;  // ignore small
  // keep order: up-left, down-left, down-right, up-right
  Point2f rect_points[4];
  bbox.points(rect_points);
  vector<Point> tmp = {rect_points[0], rect_points[1], rect_points[2], rect_points[3]};
  sort(tmp.begin(), tmp.end(), point_cmp);
  int idx_0 = 0, idx_1 = 1, idx_2 = 2, idx_3 = 3;
  if (tmp[1].y > tmp[0].y) { idx_0 = 0; idx_3 = 1; }
  else                     { idx_0 = 1; idx_3 = 0; }
  if (tmp[3].y > tmp[2].y) { idx_1 = 2; idx_2 = 3; }
  else                     { idx_1 = 3; idx_2 = 2; }
  // change inplace to save mem :)
  contour.clear();
  contour.emplace_back(tmp[idx_0]);
  contour.emplace_back(tmp[idx_1]);
  contour.emplace_back(tmp[idx_2]);
  contour.emplace_back(tmp[idx_3]);
  return false;
}
inline float box_score(vector<Point> &box, Mat &bitmap) {
  int h = bitmap.rows, w = bitmap.cols;
  int xmin = w, xmax = -1, ymin = h, ymax = -1;
  for (auto &p : box) {
    xmin = min(p.x, xmin);
    xmax = max(p.x, xmax);
    ymin = min(p.y, xmin);
    ymax = max(p.y, xmax);
  }
  xmin = clip(xmin, 0, w - 1);
  xmax = clip(xmax, 0, w - 1);
  ymin = clip(ymin, 0, h - 1);
  ymax = clip(ymax, 0, h - 1);
  int mask_h = xmax - xmin + 1, mask_w = ymax - ymin + 1;
  Mat mask = Mat::zeros(mask_h, mask_w, CV_8U);
  vector<Point> nbox;   // shift
  for (auto &p : box) {
    auto np = Point(p.x - xmin, p.y - ymin);
    nbox.emplace_back(np);
  }
  vector<vector<Point>> nboxes = { nbox };
  const Scalar color(1.0);
  fillPoly(mask, nboxes, color);
  Mat roi;
  bitmap(Rect(ymin, xmin, mask_w, mask_h)).convertTo(roi, CV_32F);
  return mean(roi, mask)[0];
}
inline void box_unclip(vector<Point> &box) {
  // calculate center point
  Point o(0, 0);
  for (auto &p : box) { o.x += p.x; o.y += p.y; }
  int n_pts = box.size();
  o.x /= n_pts; o.y /= n_pts;
  // extend away from center
  for (auto &p : box) {
    auto v = (p - o) * DET_UNCLIP_RATIO;
    p.x += v.x; p.y += v.y;
  }
}
inline Mat rotate_crop_image(Mat &im, vector<Point> &points) {
  float img_crop_width  = max(16, max(norm(points[0] - points[1]), norm(points[2] - points[3])));
  float img_crop_height = max(16, max(norm(points[0] - points[3]), norm(points[1] - points[2])));

  vector<Point2f> pts_std = {
    {0,              0}, 
    {img_crop_width, 0}, 
    {img_crop_width, img_crop_height}, 
    {0,              img_crop_height}
  };
  Mat im_sub;
  vector<Point2f> points_fp32;
  for (auto &p : points) points_fp32.emplace_back(p);
  auto M = getPerspectiveTransform(points_fp32, pts_std);
  warpPerspective(im, im_sub, M, Size(img_crop_width, img_crop_height), BORDER_REPLICATE, INTER_CUBIC);
  if (float(im_sub.rows) / im_sub.cols >= 1.5)
    rotate(im_sub, im_sub, ROTATE_90_COUNTERCLOCKWISE);
  return im_sub;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s ppocr_det.cvimodel ppocr_rec.cvimodel image.jpg\n", argv[0]);
    exit(-1);
  }

  #pragma region var
  // one-time timer
  clock_t ts_init = clock(), ts_start;
  // accumulated timer
  clock_t ts_rec_pre = 0, ts_rec_infer = 0, ts_rec_post = 0;
  // model aux vars
  CVI_MODEL_HANDLE model = nullptr;
  CVI_RC ret_code;
  CVI_TENSOR *inputs, *outputs, *input, *output;
  int32_t input_num, output_num;
  uint8_t *ptr;
  #pragma endregion var

  #pragma region img
  ts_start = clock();
  Mat im = imread(argv[3]);  // BGR-HWC
  if (!im.data) {
    printf("Could not open or find the image file: %s\n", argv[3]);
    exit(2);
  }
  printf("load image: %d clock\n", clock() - ts_start);
  #pragma endregion img

  putchar('\n');

  #pragma region det
  // model load
  ts_start = clock();
  ret_code = CVI_NN_RegisterModel(argv[1], &model);
  if (ret_code != CVI_RC_SUCCESS) { printf("CVI_NN_RegisterModel model_det failed, err %d\n", ret_code); exit(1); }
  printf("load model: %d clock\n", clock() - ts_start);

  CVI_NN_GetInputOutputTensors(model, &inputs, &input_num, &outputs, &output_num);
  input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, inputs,  input_num);
  output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, outputs, output_num);
  ptr = (uint8_t *) CVI_NN_TensorPtr(input);      // (1, 3, 640, 640), uint8
  if (DEBUG_MODEL) {
    printf("[det model info]\n");
    printf("  input->fmt: %d\n", input->fmt);       // CVI_FMT_UINT8=7
    printf("  input->count: %d\n", input->count);   // 1228800
    printf("  input->shape: %d (%d, %d, %d, %d)\n", input->shape.dim_size, input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]); // 4 (1, 640, 640, 3)
    printf("  input->pixel_format: %d\n", input->pixel_format);   // CVI_NN_PIXEL_BGR_PACKED=1
    printf("  output->fmt: %d\n", output->fmt);     // CVI_FMT_INT8=6
    printf("  output->count: %d\n", output->count); // 409600
    printf("  output->shape: %d (%d, %d, %d, %d)\n", output->shape.dim_size, output->shape.dim[0], output->shape.dim[1], output->shape.dim[2], output->shape.dim[3]); // 4 (1, 1, 640, 640)
    printf("  output->pixel_format: %d\n", output->pixel_format); // CVI_NN_PIXEL_TENSOR=100
  }

  // preprocess (resize)
  ts_start = clock();
  int W = im.cols, H = im.rows;
  int size_max = max(W, H);
  int cvs_size = min(size_max, DET_IMG_SIZE);
  int W_resize = W * cvs_size / size_max, 
      H_resize = H * cvs_size / size_max;
  if (W != W_resize || H != H_resize)
    resize(im, im, Size(W_resize, H_resize));       // default to linear interp 
  Mat cvs = Mat::zeros(cvs_size, cvs_size, CV_8UC3);
  im.copyTo(cvs(Rect(0, 0, W_resize, H_resize)));   // paste im to left-upper of cvs
  printf("ts_det_pre: %d clock\n", clock() - ts_start);

  // infer
  ts_start = clock();
  memcpy(ptr, cvs.reshape(1).data, (cvs_size * cvs_size * 3 * sizeof(uint8_t)));
  CVI_NN_Forward(model, inputs, input_num, outputs, output_num);
  int8_t *plogits = (int8_t *) CVI_NN_TensorPtr(output);  // (1, 1, 640, 640), int8
  printf("ts_det_infer: %d clock\n", clock() - ts_start);

  // postprocess (bitmap -> boxes)
  ts_start = clock();
  Mat bitmap = Mat(cvs_size, cvs_size, CV_8UC1, plogits) >= int8_t(DET_SEG_THRESH * DET_QSCALE);
#ifdef DEBUG_OUT
  imwrite("bitmap.png", bitmap);
#endif
  vector<vector<Point>> contours;
  findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
  int n_contours = min(contours.size(), DET_MAX_BOXES);
  vector<vector<Point>> boxes;
  for (int i = 0 ; i < n_contours ; i++) {
    auto box = contours[i];
    if (box.size() < 4) continue;

    bool ignore = box_bound(box);
    if (ignore) continue;
/*
    float score = box_score(box, bitmap);
    cout << "score: " << score << endl;
    if (score < DET_BOX_THRESH || score > 2) continue;
*/
    box_unclip(box);
    for (auto &p : box) {
      p.x = clip(p.x * W / W_resize, 0, W);
      p.y = clip(p.y * H / H_resize, 0, H);
    }
    //cout << box << endl;
    boxes.emplace_back(box);
  }
  printf("ts_det_post: %d clock (found %d/%d boxes)\n", clock() - ts_start, boxes.size(), n_contours);

  // model unload
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  printf("unload model: %d clock\n", clock() - ts_start);
  #pragma endregion det

  putchar('\n');

  #pragma region rec
  // model load
  ts_start = clock();
  ret_code = CVI_NN_RegisterModel(argv[2], &model);
  if (ret_code != CVI_RC_SUCCESS) { printf("CVI_NN_RegisterModel model_rec failed, err %d\n", ret_code); exit(1); }
  printf("load model: %d clock\n", clock() - ts_start);

  CVI_NN_GetInputOutputTensors(model, &inputs, &input_num, &outputs, &output_num);
  input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, inputs,  input_num );
  output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, outputs, output_num);
  ptr = (uint8_t *) CVI_NN_TensorPtr(input);        // (1, 1, 48, 640), uint8
  const size_t L = output->shape.dim[1], D = output->shape.dim[2];  // 160, 6625
  if (DEBUG_MODEL) {
    printf("[rec model info]\n");
    printf("  input->fmt: %d\n", input->fmt);       // CVI_FMT_UINT8=7
    printf("  input->count: %d\n", input->count);   // 92160
    printf("  input->shape: %d (%d, %d, %d, %d)\n", input->shape.dim_size, input->shape.dim[0], input->shape.dim[1], input->shape.dim[2], input->shape.dim[3]); // 4 (1, 48, 640, 3)
    printf("  input->pixel_format: %d\n", input->pixel_format);   // CVI_NN_PIXEL_BGR_PACKED=1
    printf("  output->fmt: %d\n", output->fmt);     // CVI_FMT_FP32=0
    printf("  output->count: %d\n", output->count); // 1060000
    printf("  output->shape: %d (%d, %d, %d, %d)\n", output->shape.dim_size, output->shape.dim[0], output->shape.dim[1], output->shape.dim[2], output->shape.dim[3]); // 4 (1, 160, 6625, 1)
    printf("  output->pixel_format: %d\n", output->pixel_format); // CVI_NN_PIXEL_TENSOR=100
  }

  for (int b = 0 ; b < boxes.size() ; b++) {
    // preprocess
    ts_start = clock();
    Mat im_sub = rotate_crop_image(im, boxes[b]);
#ifdef DEBUG_OUT
    char filename[100];
    sprintf(filename, "crop_box-%d.png", b);
    imwrite(filename, im_sub);
#endif
    W = im_sub.cols; H = im_sub.rows;       // fix H to 48
    float ratio = float(H) / REC_IMG_HEIGHT;
    H_resize = REC_IMG_HEIGHT;
    W_resize = W / ratio;
    if (W != W_resize || H != H_resize)
      resize(im_sub, im_sub, Size(W_resize, H_resize));
    cvs = Mat::zeros(H_resize, W_resize, CV_8UC3);
    im_sub.copyTo(cvs(Rect(0, 0, W_resize, H_resize)));
    ts_rec_pre += clock() - ts_start;

    // infer
    ts_start = clock();
    memcpy(ptr, cvs.reshape(1).data, (H_resize * W_resize * 3 * sizeof(uint8_t)));
    CVI_NN_Forward(model, inputs, input_num, outputs, output_num);
    float_t *logits = (float_t *) CVI_NN_TensorPtr(output);  // (1, 160, 6625, 1), fp32
    ts_rec_infer += clock() - ts_start;

    // postprocess
    ts_start = clock();
    float logit_max, logit;
    int offset, idx, last_idx = -1;
    printf("[Box-%d]\n", b);
    cout << boxes[b] << endl;
    printf("ids: ");
    for (int k = 0 ; k < L; k++) {
      logit_max = REC_LOGIT_NINF, idx = -1;
      offset = k * D;
      for (int i = 0 ; i < D; i++) {
        logit = logits[offset + i];
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
    ts_rec_post += clock() - ts_start;
  }
  printf("ts_rec_pre: %d clock\n",   ts_rec_pre);
  printf("ts_rec_infer: %d clock\n", ts_rec_infer);
  printf("ts_rec_post: %d clock\n",  ts_rec_post);

  // model unload
  ts_start = clock();
  CVI_NN_CleanupModel(model);
  printf("unload model: %d clock\n", clock() - ts_start);
  #pragma endregion rec

  putchar('\n');

  printf("Total time cost: %d clock\n", clock() - ts_init);
  printf("CLOCKS_PER_SEC: %d\n", CLOCKS_PER_SEC);
}
