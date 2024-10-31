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

// ppocr pipeline: det + rec
// TODO: do NOT use clock() for timer!!

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
#define DET_MIN_SIZE      3
#define DET_UNCLIP_K      3.0
#define DET_UNCLIP_T      1.414
#define DET_QSCALE        127
#define REC_IMG_WIDTH     320
#define REC_IMG_HEIGHT    32
#define REC_LOGIT_NINF    -114514191981.0

//#define DEBUG_DUMP_IMG
#pragma endregion def


int main(int argc, char *argv[]) {
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
  printf("imgs.shape: w=%d, h=%d\n", im.cols, im.rows);
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
    im.copyTo(cvs(Rect(0, 0, W_resize, H_resize))); // paste im to left-upper of cvs
  } else {
    im.copyTo(cvs(Rect(0, 0, W, H)));
  }
  printf("ts_det_pre: %d clock\n", clock() - ts_start);

  // infer
  ts_start = clock();
  memcpy(ptr, cvs.reshape(1).data, cvs.step[0] * cvs.rows);
  CVI_NN_Forward(model, inputs, input_num, outputs, output_num);
  int8_t *plogits = (int8_t *) CVI_NN_TensorPtr(output);
  printf("ts_det_infer: %d clock\n", clock() - ts_start);

  // postprocess (bitmap -> boxes)
  ts_start = clock();
  Mat bitmap = Mat(cvs_size, cvs_size, CV_8SC1, plogits) >= int8_t(DET_SEG_THRESH * DET_QSCALE);   // vset {0, 255}

#ifdef DEBUG_DUMP_IMG
  imwrite("bitmap.png", bitmap);
#endif

  vector<Contour> contours;
  findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
  int n_contours = min(contours.size(), DET_MAX_BOXES);
  vector<Box> boxes;
  for (int i = 0 ; i < n_contours ; i++) {
    auto contour = contours[i];
    if (contour.size() < 4) continue;
    // bouding box
    RotatedRect bbox = minAreaRect(contour);
    int w = bbox.size.width, h = bbox.size.height;
    //cout << "bbox size: w=" << w << " h=" << h << endl;
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
  ptr = (uint8_t *) CVI_NN_TensorPtr(input);
  const size_t L = output->shape.dim[1], D = output->shape.dim[2];

  for (int b = 0 ; b < boxes.size() ; b++) {
    Box &box = boxes[b];

    // preprocess
    ts_start = clock();
    vector<Point2f> pts;  // dtype cvt
    for (auto &p : box) pts.emplace_back(p);
    float img_crop_width  = max(16, max(norm(box[0] - box[1]), norm(box[2] - box[3])));
    float img_crop_height = max(16, max(norm(box[0] - box[3]), norm(box[1] - box[2])));
    vector<Point2f> pts_std = {
      {0,              0}, 
      {img_crop_width, 0}, 
      {img_crop_width, img_crop_height}, 
      {0,              img_crop_height}
    };
    auto M = getPerspectiveTransform(pts, pts_std);
    Mat im_crop;
    warpPerspective(im, im_crop, M, Size(img_crop_width, img_crop_height), INTER_CUBIC, BORDER_REPLICATE);
    if (float(im_crop.rows) / im_crop.cols >= 1.5)
      rotate(im_crop, im_crop, ROTATE_90_COUNTERCLOCKWISE);

#ifdef DEBUG_DUMP_IMG
    char filename[32];
    sprintf(filename, "crop_box-%d.png", b);
    imwrite(filename, im_crop);
#endif

    int W = im_crop.cols, H = im_crop.rows;
    int W_resize = min(W * REC_IMG_HEIGHT / H, REC_IMG_WIDTH);
    if (W != W_resize || H != REC_IMG_HEIGHT)
      resize(im_crop, im_crop, Size(W_resize, REC_IMG_HEIGHT));
    Mat cvs(REC_IMG_HEIGHT, REC_IMG_WIDTH, CV_8UC3, Scalar::all(255));
    im_crop.copyTo(cvs(Rect(0, 0, W_resize, REC_IMG_HEIGHT)));
    ts_rec_pre += clock() - ts_start;

#ifdef DEBUG_DUMP_IMG
    sprintf(filename, "crop_box_cvs-%d.png", b);
    imwrite(filename, cvs);
#endif

    // infer
    ts_start = clock();
    memcpy(ptr, cvs.reshape(1).data, cvs.step[0] * cvs.rows);
    CVI_NN_Forward(model, inputs, input_num, outputs, output_num);
    float_t *logits = (float_t *) CVI_NN_TensorPtr(output);
    ts_rec_infer += clock() - ts_start;

    // postprocess
    ts_start = clock();
    printf("[Box-%d]\n", b);
    cout << boxes[b] << endl;
    printf("ids: ");
    int last_idx = -1;
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

  clock_t ts_total = clock() - ts_init;
  printf("Total time cost: %d clock (%.3f ms)\n", ts_total, double(ts_total) * 1000 / CLOCKS_PER_SEC);
  printf("CLOCKS_PER_SEC: %d\n", CLOCKS_PER_SEC);
}
