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
#include <iostream>
#include <tuple>
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
//constexpr char kOutputStream[] = "output_video";
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

// mediapipe 그래프 호출
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
  // output: 비디오 대신 face landmarks을 return
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

// model img와 src img의 landmark의 위치 세팅
// 을 위한 coefficient 행렬 구하기
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
// 받아온 행렬값으로 scaling, translation 연산
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

// RBF(Radial Basis Funtion) 커널 선택
// 왜곡이 극단적이라 효과가 눈에 띄지만 배경 왜곡이 심함
float thinPlateSpline(const Point2f& p1, const Point2f& p2) {
  auto d = sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
  if (d == 0) return 1;
  return d*d*std::log(d);
}
// 자연스러운 왜곡
float gaussian(const Point2f& p1, const Point2f& p2){
  auto d = sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
  auto sigma = 2;
  return exp(-d / (2 * sigma * sigma));
}

// scr landmark의 분포를 model landmark로 근사
// 을 위해 dx, dy 기반으로 weight 계산
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
// 받아온 weight값으로 src이미지 변환 map 계산
std::tuple<Mat,Mat> RBF(const std::vector<Point2f>& src_landmarks, const Mat& srcImg, const std::tuple<Mat,Mat>& weight){
  int ratio=2;
  Mat map_x = Mat::zeros(srcImg.rows/ratio,srcImg.cols/ratio, CV_32F);
  Mat map_y = Mat::zeros(srcImg.rows/ratio,srcImg.cols/ratio, CV_32F);
  for (int row = 0; row < map_x.rows; row++){
    for (int col = 0; col < map_x.cols; col++){
      for (int k = 0; k < src_landmarks.size(); k++){
        //map_x.at<float>(row,col) -= std::get<0>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(row*ratio,col*ratio)); 
        //map_y.at<float>(row,col) -= std::get<1>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(col*ratio,row*ratio));
        map_x.at<float>(row,col) -= std::get<0>(weight).at<float>(k)*gaussian(src_landmarks[k], Point2f(row*ratio,col*ratio)); 
        map_y.at<float>(row,col) -= std::get<1>(weight).at<float>(k)*gaussian(src_landmarks[k], Point2f(col*ratio,row*ratio));
      }
      map_x.at<float>(row,col) += col*ratio;
      map_y.at<float>(row,col) += row*ratio;
    }
  }
  resize(map_x,map_x,srcImg.size());
  resize(map_y,map_y,srcImg.size());
  return std::make_tuple(map_x, map_y);
}

// GUI를 위한 변수 설정
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;
Mat src1;
Mat dst;
Mat map_x2;
Mat map_y2;
 
// Trackbar에 대한 콜백 함수
static void on_trackbar( int, void* ){
  alpha = (double) alpha_slider/alpha_slider_max ;
  beta = ( 1.0 - alpha );
  Mat map_x3(src1.size(),CV_32FC1);
  Mat map_y3(src1.size(),CV_32FC1);
  for (int i = 0; i < src1.rows; i++){
    for (int j = 0; j < src1.cols; j++){
    map_x3.at<float>(i,j) = alpha * map_x2.at<float>(i,j) + beta * j;
    map_y3.at<float>(i,j) = alpha * map_y2.at<float>(i,j) + beta * i;
    }
  }
  remap(src1, dst, map_x3, map_y3, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
  imshow( "Blend", dst );
}

// main
int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  for( int i=0;i<argc;i++) printf("%s\n",argv[i]);

  Mat srcImg = imread(argv[2]);
  Mat src2 = imread(argv[3]);
  
  imshow("src", srcImg);
  imshow("model", src2);  
  waitKey(1);
  
  mediapipe::NormalizedLandmarkList landmarks;
  mediapipe::NormalizedLandmarkList landmarks2;
  RunMPPGraph(srcImg, landmarks);
  RunMPPGraph(src2, landmarks2);
  std::vector<Point2f> p1;
  std::vector<Point2f> p2;

  for (int i = 0; i < landmarks.landmark_size(); i++){
    p1.push_back(Point2f((float)landmarks.landmark(i).x()*srcImg.cols,(float)landmarks.landmark(i).y()*srcImg.rows));
    p2.push_back(Point2f((float)landmarks2.landmark(i).x()*src2.cols,(float)landmarks2.landmark(i).y()*src2.rows));
    // 첫 번째 이미지에 랜드마크 빨간 점으로 표시
    //circle(srcImg, Point(landmarks.landmark(i).x()*srcImg.cols, landmarks.landmark(i).y()*srcImg.rows), 1, Scalar(0, 0, 255), - 1);
    // 두 번째 이미지에 랜드마크 파란 점으로 표시
    //circle(src2, Point(landmarks2.landmark(i).x()*src2.cols, landmarks2.landmark(i).y()*src2.rows), 1, Scalar(255, 0, 0), - 1);
  }

  // src와 model 이미지의 랜드마크 분포 비교를 위해 (src이미지에 대한) transformation 과정 : srcImg -> src1
  auto coeff = getLinearCoeff(p2, p1); 
  auto [map_x, map_y] = SLR(src2, coeff);
  remap(srcImg, src1, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
  // transform된 소스 이미지(src1)에 대해 landmark를 다시 계산
  mediapipe::NormalizedLandmarkList landmarks3;
  RunMPPGraph(src1, landmarks3);
  std::vector<Point2f> p3;
  for (int i = 0; i < landmarks.landmark_size(); i++){
    p3.push_back(Point2f((float)landmarks3.landmark(i).x()*src1.cols,(float)landmarks3.landmark(i).y()*src1.rows));
  }

  // src1의 landmark의 분포를 model의 landmark의 분포로 근사
  auto weight = getRBFWeight(p3, p2);
  std::tuple<Mat,Mat> result = RBF(p3, src1, weight);
  map_x2 = std::get<0>(result);
  map_y2 = std::get<1>(result);

  // 결과 이미지 - 처음 이미지는 before이므로 src1 img를 show
  dst = src1.clone();
  imshow("Blend", dst); 
  // Trackbar로 map의 값을 조정
  createTrackbar( "g_sigma", "Blend", &alpha_slider, alpha_slider_max, on_trackbar );
  waitKey();

  return EXIT_SUCCESS;
}

// 명령 위치
// ~/mediapipe_repo/mediapipe
// 빌드 명령
// bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --action_env PYTHON_BIN_PATH="C://Python//python.exe" mediapipe/examples/desktop/face_mesh:face_deformation_gaussian
// 실행 명령(예시), .pptxt 뒤는 순서대로 src이미지 경로, model이미지 경로
// GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/face_mesh/face_deformation_gaussian --calculator_graph_config_file=mediapipe/graphs/face_mesh/face_mesh_desktop_live.pbtxt C:/dh.jpg C:/ya.jpg

// hm -> ys