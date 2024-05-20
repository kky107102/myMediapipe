// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cstdlib>
#include <cmath>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_calib3d_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_features2d_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"

using namespace cv;
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
constexpr char kLandmarksStream[] = "single_face_landmarks";
constexpr char kROIStream[] = "face_rect_from_landmarks";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph(Mat img, mediapipe::NormalizedLandmarkList& output_landmarks) {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);
  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  ABSL_LOG(INFO) << "Start running the calculator graph.";
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_single_landmark, graph.AddOutputStreamPoller(kLandmarksStream));
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  cv::Mat inputImg;
  cv::cvtColor(img, inputImg, cv::COLOR_BGR2RGB);
  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, inputImg.cols, inputImg.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  inputImg.copyTo(input_frame_mat);

  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));
  mediapipe::Packet landmark_packet;
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  if (!poller_single_landmark.Next(&landmark_packet)) ABSL_LOG(INFO) << "stop graph";
  output_landmarks = landmark_packet.Get<mediapipe::NormalizedLandmarkList>();
  return graph.WaitUntilDone();
}     

Mat getLinearCoeff(const std::vector<Point2f>& src_landmarks, const std::vector<Point2f>& model_landmarks){
  int lmsize = src_landmarks.size();
  Mat modelPoints(lmsize, 2, CV_32F);
  Mat srcPoints(lmsize, 3, CV_32F);
  Mat coeff(2, 3, CV_32F);

  for (int i = 0; i < lmsize; i++){
    modelPoints.at<float>(i,0) = model_landmarks[i].x;
    modelPoints.at<float>(i,1) = model_landmarks[i].y;
    srcPoints.at<float>(i,0) = src_landmarks[i].x;
    srcPoints.at<float>(i,1) = src_landmarks[i].y;
    srcPoints.at<float>(i,2) = (float)1;
  }
  
  solve(srcPoints, modelPoints, coeff, DECOMP_SVD);
  return coeff;
}

std::tuple<Mat,Mat> SLR(const Mat& dstImg, const Mat& coeff){
  Mat map_x = Mat::zeros(dstImg.size(), CV_32F);
  Mat map_y = Mat::zeros(dstImg.size(), CV_32F);
  
  for (int row = 0; row < dstImg.rows; row++){
    for (int col = 0; col < dstImg.cols; col++){
      map_x.at<float>(row,col) += (col * coeff.at<float>(0,0) + row * coeff.at<float>(1,0) + coeff.at<float>(2,0)); //x
      map_y.at<float>(row,col) += (col * coeff.at<float>(0,1) + row * coeff.at<float>(1,1) + coeff.at<float>(2,1)); //y
    }
  }
  return std::make_tuple(map_x, map_y);
}

float thinPlateSpline(const Point2f& p1, const Point2f& p2) {
  auto d = sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
  if (d == 0) return 1;
  return d*d*log(d);
}

float gaussian(const Point2f& p1, const Point2f& p2){
  auto d = sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
  auto sigma = 5;
  return exp(-d / (2 * sigma * sigma));
}

std::tuple<Mat,Mat> getRBFWeight(const std::vector<Point2f>& src_landmarks, const std::vector<Point2f>& model_landmarks) {
  int lmsize = src_landmarks.size();
  Mat del_x(lmsize, 1, CV_32F);
  Mat del_y(lmsize, 1, CV_32F);
  Mat matrix(lmsize, lmsize, CV_32F);
  Mat weight_x(lmsize,1,CV_32F);
  Mat weight_y(lmsize,1,CV_32F);

  for(int i = 0; i < lmsize; i++){
    del_x.at<float>(i) = model_landmarks[i].x - src_landmarks[i].x;
    del_y.at<float>(i) = model_landmarks[i].y - src_landmarks[i].y;
    for (int j = 0; j < lmsize; j++){
      //matrix.at<float>(i,j) = thinPlateSpline(src_landmarks[j], src_landmarks[i]);
      matrix.at<float>(i,j) = gaussian(src_landmarks[j], src_landmarks[i]);
    }
  }

  solve(matrix, del_x, weight_x);  
  solve(matrix, del_y, weight_y);

  return std::make_tuple(weight_x, weight_y);
}

std::tuple<Mat,Mat> RBF(const std::vector<Point2f>& src_landmarks, const Mat& srcImg, const std::tuple<Mat,Mat>& weight){
  Mat map_x = Mat::zeros(srcImg.size(), CV_32F);
  Mat map_y = Mat::zeros(srcImg.size(), CV_32F);
  for (int row = 0; row < srcImg.rows; row++){
    for (int col = 0; col < srcImg.cols; col++){
      for (int k = 0; k < src_landmarks.size(); k++){
        // map_x.at<float>(row,col) -= std::get<0>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(row,col)); 
        // map_y.at<float>(row,col) -= std::get<1>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(col,row));
        map_x.at<float>(row,col) -= std::get<0>(weight).at<float>(k)*gaussian(src_landmarks[k], Point2f(row,col)); 
        map_y.at<float>(row,col) -= std::get<1>(weight).at<float>(k)*gaussian(src_landmarks[k], Point2f(col,row));
      }
      map_x.at<float>(row,col) += col;
      map_y.at<float>(row,col) += row;
    }
  }
  return std::make_tuple(map_x, map_y);
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  Mat srcImg = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/jehoon.jpg");
  Mat modelImg = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/jehoon.jpg");
  mediapipe::NormalizedLandmarkList landmarks;
  mediapipe::NormalizedLandmarkList landmarks2;
  RunMPPGraph(srcImg, landmarks);
  RunMPPGraph(modelImg, landmarks2);
  std::vector<Point3f> p1;
  std::vector<Point3f> p2;

  for (int i = 0; i < landmarks.landmark_size(); i++){
    p1.push_back(Point3f((float)landmarks.landmark(i).x()*srcImg.cols,(float)landmarks.landmark(i).y()*srcImg.rows,(float)landmarks.landmark(i).z()));
    std::cout << i+1 << " "<< (float)landmarks.landmark(i).z() << std::endl;
    p2.push_back(Point3f((float)landmarks2.landmark(i).x()*modelImg.cols,(float)landmarks2.landmark(i).y()*modelImg.rows,(float)landmarks2.landmark(i).z()));
    // 첫 번째 이미지에 랜드마크 빨간 점으로 표시
    circle(srcImg, Point(landmarks.landmark(i).x()*srcImg.cols, landmarks.landmark(i).y()*srcImg.rows), 1, Scalar(0, 0, 255), - 1);
    // 두 번째 이미지에 랜드마크 파란 점으로 표시
    circle(modelImg, Point(landmarks2.landmark(i).x()*modelImg.cols, landmarks2.landmark(i).y()*modelImg.rows), 1, Scalar(255, 0, 0), - 1);
  }
//   Mat H = findHomography(p1, p2);
//   Mat imgwarp;
//   warpPerspective(srcImg, imgwarp, H, Size(srcImg.cols*1.5 , srcImg.rows *1.5));
//   std::vector<Point2f> pointwarp;
//   perspectiveTransform(p1, pointwarp, H); 
//   for (int i = 0; i < landmarks.landmark_size(); i++){
//     // 두 번째 이미지에 첫번째 랜드마크 빨간 점으로 표시
//     circle(modelImg, pointwarp[i], 1, Scalar(0, 0, 225), - 1);
//   }
//   auto coeff = getLinearCoeff(p2, p1); //?
//   auto [map_x, map_y] = SLR(modelImg, coeff);
//   Mat similarityImg(srcImg.size() , srcImg.type());
//   remap(srcImg, similarityImg, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));

//   mediapipe::NormalizedLandmarkList landmarks3;
//   RunMPPGraph(similarityImg, landmarks3);
//   std::vector<Point2f> p3;
//   for (int i = 0; i < landmarks.landmark_size(); i++){
//     p3.push_back(Point2f((float)landmarks3.landmark(i).x()*similarityImg.cols,(float)landmarks3.landmark(i).y()*similarityImg.rows));
//     circle(modelImg, p3[i], 1, Scalar(0, 0, 225), - 1);
//   }

//   Mat dst(srcImg.size() , srcImg.type());
//   auto weight = getRBFWeight(p3, p2);
//   auto [map_x1, map_y2] = RBF(p3, similarityImg, weight);
//   remap(similarityImg, dst, map_x1, map_y2, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));

//   imshow("similarity", similarityImg);
     imshow("ms", srcImg);
//   imshow("model", modelImg);  
//   imshow("Similarity + RBF(gaussian)", dst); 
   imwrite("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/ms_point.jpg", srcImg);
   waitKey();

  return EXIT_SUCCESS;
}