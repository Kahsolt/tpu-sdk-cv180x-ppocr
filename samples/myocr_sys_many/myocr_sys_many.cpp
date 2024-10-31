#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <cviruntime.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// my ocr pipeline: ppocr det + chocr rec
// https://blog.csdn.net/u012351051/article/details/109372643

#pragma region def
typedef vector<Point> Contour;
typedef vector<Point> Box;

#define min(x, y) (((x) <= (y)) ? (x) : (y))
#define max(x, y) (((x) >= (y)) ? (x) : (y))
#define clip(x, a, b) min(max((x), (a)), (b))
#define us_to_ms(x) (double(x) / 1000)

#define DET_IMG_SIZE      640
#define DET_SEG_THRESH    0.6
#define DET_MAX_BOXES     10
#define DET_MIN_SIZE      7
#define DET_UNCLIP_K      2.7
#define DET_UNCLIP_T      1.414
#define DET_QSCALE        127
#define REC_IMG_WIDTH     320
#define REC_IMG_HEIGHT    32
#define REC_LOGIT_NINF    -114514191981.0

#define SAVE_FILE_PATH    "results.txt"
#define SAVE_DIR_PATH     "results"
#define FILE_PATH_MAXLEN  256

//#define DEBUG_DUMP_DET
//#define DEBUG_DUMP_REC
#pragma endregion def

inline float timeval_to_ms(struct timeval &a, struct timeval &b) {
  float a_ms = a.tv_sec * 1000 + float(a.tv_usec) / 1000;
  float b_ms = b.tv_sec * 1000 + float(b.tv_usec) / 1000;
  return b_ms - a_ms;
}
inline time_t timeval_to_us(struct timeval &a, struct timeval &b) {
  time_t a_ns = a.tv_sec * 1000000 + a.tv_usec;
  time_t b_ns = b.tv_sec * 1000000 + b.tv_usec;
  return b_ns - a_ns;
}
int point_cmp(const void *a, const void *b) {
  Point2f *pa = (Point2f *) a, *pb = (Point2f *) b;
  return pa->x != pb->x ? pa->x - pb->x : pa->y - pb->y;
}

int main(int argc, char *argv[]) {
  #pragma region chk
  if (argc != 4) {
    printf("Usage: %s ppocr_det.cvimodel chocr_rec.cvimodel image_folder\n", argv[0]);
    exit(-1);
  }

  char *base_path = argv[3];
  DIR *dir = opendir(base_path);
  if (!dir) {
    printf("Can not opendir() %s\n", base_path);
    exit(3);
  }

#if defined(DEBUG_DUMP_DET) || defined(DEBUG_DUMP_REC)
  char dump_fp[FILE_PATH_MAXLEN];
  if (!opendir(SAVE_DIR_PATH))
    mkdir(SAVE_DIR_PATH, 0755);
#endif
  #pragma endregion chk

  #pragma region stats
  int n_img = 0, n_crop = 0;
  struct timeval tv_init, tv_start, tv_end;
  gettimeofday(&tv_init, NULL);
  // timespan in microsecond (us)
  time_t ts_det_pre  = 0, ts_det_infer = 0, ts_det_post = 0;
  time_t ts_img_load = 0, ts_img_crop  = 0;
  time_t ts_rec_pre  = 0, ts_rec_infer = 0, ts_rec_post = 0;
  #pragma endregion stats

  #pragma region load
  gettimeofday(&tv_start, NULL);

  CVI_MODEL_HANDLE det_model = nullptr, rec_model = nullptr;
  CVI_RC ret_code;
  ret_code = CVI_NN_RegisterModel(argv[1], &det_model);
  if (ret_code != CVI_RC_SUCCESS) { printf("CVI_NN_RegisterModel model_det failed, err %d\n", ret_code); exit(1); }
  ret_code = CVI_NN_RegisterModel(argv[2], &rec_model);
  if (ret_code != CVI_RC_SUCCESS) { printf("CVI_NN_RegisterModel model_rec failed, err %d\n", ret_code); exit(2); }

  CVI_TENSOR *det_inputs, *det_outputs;
  int32_t det_input_num, det_output_num;
  CVI_NN_GetInputOutputTensors(det_model, &det_inputs, &det_input_num, &det_outputs, &det_output_num);
  CVI_TENSOR *det_input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, det_inputs,  det_input_num);
  CVI_TENSOR *det_output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, det_outputs, det_output_num);
  uint8_t *det_ptr = (uint8_t *) CVI_NN_TensorPtr(det_input);

  CVI_TENSOR *rec_inputs, *rec_outputs;
  int32_t rec_input_num, rec_output_num;
  CVI_NN_GetInputOutputTensors(rec_model, &rec_inputs, &rec_input_num, &rec_outputs, &rec_output_num);
  CVI_TENSOR *rec_input  = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, rec_inputs,  rec_input_num);
  CVI_TENSOR *rec_output = CVI_NN_GetTensorByName(CVI_NN_DEFAULT_TENSOR, rec_outputs, rec_output_num);
  uint8_t *rec_ptr = (uint8_t *) CVI_NN_TensorPtr(rec_input);
  const size_t L = rec_output->shape.dim[0], D = rec_output->shape.dim[2];

  gettimeofday(&tv_end, NULL);
  printf("ts_model_load: %.3f ms\n", timeval_to_ms(tv_start, tv_end));
  #pragma endregion load

  #pragma region process
  // recyclable vars
  Mat im, im_crop, cvs, bitmap;
  vector<Contour> contours;
  vector<Box> boxes;
  vector<Point2f> pts;
  // fs & write
  FILE* fout = fopen(SAVE_FILE_PATH, "w");
  struct dirent *entry;
  char fp[FILE_PATH_MAXLEN];
  printf(">> progress: ");
  while (entry = readdir(dir)) {
    if (entry->d_name[0] == '.') continue;

    // write
    memset(fp, 0x00, sizeof(fp));
    strcpy(fp, base_path);
    strcat(fp, "/");
    strcat(fp, entry->d_name);
    fprintf(fout, "%s\n", entry->d_name);
    putchar('.'); fflush(stdout);
    n_img++;

    #pragma region img
    gettimeofday(&tv_start, NULL);
    im = imread(fp);
    gettimeofday(&tv_end, NULL);
    ts_img_load += timeval_to_us(tv_start, tv_end);
    if (!im.data) {
      printf("Could not open or find the image file: %s\n", fp);
      continue;
    }
    #pragma endregion img

    #pragma region det
    // preprocess (resize)
    gettimeofday(&tv_start, NULL);
    int W = im.cols, H = im.rows;
    int size_max = max(W, H);
    cvs = Mat::zeros(DET_IMG_SIZE, DET_IMG_SIZE, CV_8UC3);
    double det_r = double(size_max) / DET_IMG_SIZE;
    if (size_max > DET_IMG_SIZE) {
      int W_resize = W / det_r, 
          H_resize = H / det_r;
      resize(im, im, Size(W_resize, H_resize));   // inplace!
      im.copyTo(cvs(Rect(0, 0, W_resize, H_resize)));
    } else {
      im.copyTo(cvs(Rect(0, 0, W, H)));
    }
    gettimeofday(&tv_end, NULL);
    ts_det_pre += timeval_to_us(tv_start, tv_end);

    // infer
    gettimeofday(&tv_start, NULL);
    memcpy(det_ptr, cvs.reshape(1).data, cvs.step[0] * cvs.rows);
    CVI_NN_Forward(det_model, det_inputs, det_input_num, det_outputs, det_output_num);
    int8_t *plogits = (int8_t *) CVI_NN_TensorPtr(det_output);
    gettimeofday(&tv_end, NULL);
    ts_det_infer += timeval_to_us(tv_start, tv_end);

    // postprocess (bitmap -> boxes)
    gettimeofday(&tv_start, NULL);
    bitmap = Mat(DET_IMG_SIZE, DET_IMG_SIZE, CV_8SC1, plogits) >= int8_t(DET_SEG_THRESH * DET_QSCALE);

#ifdef DEBUG_DUMP_DET
    sprintf(dump_fp, "%s/bitmap-%d.png\0", SAVE_DIR_PATH, n_img);
    imwrite(dump_fp, bitmap);
#endif

    contours.clear();
    findContours(bitmap, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    int n_contours = min(contours.size(), DET_MAX_BOXES);
    boxes.clear();
    for (int i = 0 ; i < n_contours ; i++) {
      auto contour = contours[i];
      if (contour.size() < 4) continue;
      // contour to rect box
      RotatedRect bbox = minAreaRect(contour);
      int w = bbox.size.width, h = bbox.size.height;
      if (min(w, h) < DET_MIN_SIZE) continue;
      Point2f tmp[4]; bbox.points(tmp); // native order: bottomLeft, topLeft, topRight, bottomRight, but not stable
      // sorted order: top-left, top-right, bottom-right, bottom-left
      qsort(tmp, 4, sizeof(Point2f), point_cmp);
      int idx_0 = 0, idx_1 = 1, idx_2 = 2, idx_3 = 3;
      if (tmp[1].y > tmp[0].y) { idx_0 = 0; idx_3 = 1; }
      else                     { idx_0 = 1; idx_3 = 0; }
      if (tmp[3].y > tmp[2].y) { idx_1 = 2; idx_2 = 3; }
      else                     { idx_1 = 3; idx_2 = 2; }
      Box box = { tmp[idx_0], tmp[idx_1], tmp[idx_2], tmp[idx_3] };
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
    gettimeofday(&tv_end, NULL);
    ts_det_post += timeval_to_us(tv_start, tv_end);
    #pragma endregion det

    #pragma region rec
    for (int b = 0 ; b < boxes.size() ; b++) {
      Box &box = boxes[b];
      n_crop++;

      // crop
      gettimeofday(&tv_start, NULL);
      pts.clear();
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
      warpPerspective(im, im_crop, M, Size(img_crop_width, img_crop_height), INTER_CUBIC, BORDER_REPLICATE);
      if (float(im_crop.rows) / im_crop.cols >= 1.5)
        rotate(im_crop, im_crop, ROTATE_90_COUNTERCLOCKWISE);
      gettimeofday(&tv_end, NULL);
      ts_img_crop += timeval_to_us(tv_start, tv_end);

#ifdef DEBUG_DUMP_REC
      sprintf(dump_fp, "%s/crop-%d.png\0", SAVE_DIR_PATH, n_crop);
      imwrite(dump_fp, im_crop);
#endif

      // preprocess
      gettimeofday(&tv_start, NULL);
      int W_crop = im_crop.cols, H_crop = im_crop.rows;
      int W_resize = min(W_crop * REC_IMG_HEIGHT / H_crop, REC_IMG_WIDTH);
      if (W_crop != W_resize || H_crop != REC_IMG_HEIGHT)
        resize(im_crop, im_crop, Size(W_resize, REC_IMG_HEIGHT));
      cvs = Mat(REC_IMG_HEIGHT, REC_IMG_WIDTH, CV_8UC3, Scalar::all(255));
      im_crop.copyTo(cvs(Rect(0, 0, W_resize, REC_IMG_HEIGHT)));
      gettimeofday(&tv_end, NULL);
      ts_rec_pre += timeval_to_us(tv_start, tv_end);

      // infer
      gettimeofday(&tv_start, NULL);
      memcpy(rec_ptr, cvs.reshape(1).data, cvs.step[0] * cvs.rows);
      CVI_NN_Forward(rec_model, rec_inputs, rec_input_num, rec_outputs, rec_output_num);
      float_t *logits = (float_t *) CVI_NN_TensorPtr(rec_output);
      gettimeofday(&tv_end, NULL);
      ts_rec_infer += timeval_to_us(tv_start, tv_end);

      // box rescale back & write (NOTE: `box` is not used after this)
      if (size_max > DET_IMG_SIZE) {
        for (auto &p : box) {
          p.x *= det_r;
          p.y *= det_r;
        }
      }
      for (auto &p : box)
        fprintf(fout, "%d %d ", p.x, p.y);
      fputc('|', fout);

      // postprocess & write
      gettimeofday(&tv_start, NULL);
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
          fprintf(fout, " %d", idx);
          last_idx = idx;
        }
      }
      fputc('\n', fout);
      gettimeofday(&tv_end, NULL);
      ts_rec_post += timeval_to_us(tv_start, tv_end);
    }
    #pragma endregion rec

    // write
    fputc('\n', fout);
    fflush(fout);
  }
  closedir(dir);
  fflush(fout); fclose(fout);
  putchar('\n');
  #pragma endregion process

  #pragma region unload
  gettimeofday(&tv_start, NULL);
  CVI_NN_CleanupModel(det_model);
  CVI_NN_CleanupModel(rec_model);
  gettimeofday(&tv_end, NULL);
  printf("ts_model_unload: %.3f ms\n", timeval_to_ms(tv_start, tv_end));
  #pragma endregion unload

  #pragma region perfcnt
  puts("================================");
  printf("n_img:        %d\n", n_img);
  printf("n_crop:       %d\n", n_crop);
  puts("--------------------------------");
  printf("ts_img_load:  %.3f ms\n", us_to_ms(ts_img_load));
  printf("ts_img_crop:  %.3f ms\n", us_to_ms(ts_img_crop));
  printf("ts_det_pre:   %.3f ms\n", us_to_ms(ts_det_pre));
  printf("ts_det_infer: %.3f ms\n", us_to_ms(ts_det_infer));
  printf("ts_det_post:  %.3f ms\n", us_to_ms(ts_det_post));
  printf("ts_rec_pre:   %.3f ms\n", us_to_ms(ts_rec_pre));
  printf("ts_rec_infer: %.3f ms\n", us_to_ms(ts_rec_infer));
  printf("ts_rec_post:  %.3f ms\n", us_to_ms(ts_rec_post));
  puts("--------------------------------");
  printf("ts_avg_pre:   %.3f ms\n", us_to_ms(ts_det_pre   + ts_rec_pre  ) / n_img);
  printf("ts_avg_infer: %.3f ms\n", us_to_ms(ts_det_infer + ts_rec_infer) / n_img);
  printf("ts_avg_post:  %.3f ms\n", us_to_ms(ts_det_post  + ts_rec_post ) / n_img);
  puts("================================");
  printf("Total time:   %.3f ms\n", timeval_to_ms(tv_init, tv_end));
  #pragma endregion perfcnt
}
