#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class DataTransformer {
 public:
  explicit DataTransformer(const TransformationParameter& param)
    : param_(param) {
    phase_ = Caffe::phase();
  }
  virtual ~DataTransformer() {}

  void InitRand();
  void FillInOffsets(int *w, int *h, int width, int height, int crop_size) {
    w[0] = 0; h[0] = 0;
    w[1] = 0; h[1] = height - crop_size;
    w[2] = width - crop_size; h[2] = 0;
    w[3] = width - crop_size; h[3] = height - crop_size;
    w[4] = (width - crop_size) / 2; h[4] = (height - crop_size) / 2;
  }
  /**
   * @brief Applies the transformation defined in the data layer's
   * transform_param block to the data.
   *
   * @param batch_item_id
   *    Datum position within the batch. This is used to compute the
   *    writing position in the top blob's data
   * @param datum
   *    Datum containing the data to be transformed.
   * @param mean
   * @param transformed_data
   *    This is meant to be the top blob's data. The transformed data will be
   *    written at the appropriate place within the blob's data.
   */
  void Transform(const int batch_item_id, const Datum& datum,
                 const Dtype* mean, Dtype* transformed_data);
  void Transform(const int batch_item_id, const IplImage *img,
                 const Dtype* mean, Dtype* transformed_data);
 protected:
  virtual unsigned int Rand();

  // Tranformation parameters
  TransformationParameter param_;


  shared_ptr<Caffe::RNG> rng_;
  Caffe::Phase phase_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_

