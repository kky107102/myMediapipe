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
  //MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller(kOutputStream));
  //MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_rect, graph.AddOutputStreamPoller(kROIStream));
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
  //mediapipe::Packet rect_packet;
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  if (!poller_single_landmark.Next(&landmark_packet)) ABSL_LOG(INFO) << "stop graph";
  //if (!poller_rect.Next(&rect_packet)) ABSL_LOG(INFO) << "stop graph";
  output_landmarks = landmark_packet.Get<mediapipe::NormalizedLandmarkList>();
  //output_rect = rect_packet.Get<mediapipe::NormalizedRect>();
  //cv::Point2f center(output_rect.x_center()*w, output_rect.y_center()*h);
  //cv::Size2f size(output_rect.width()*w, output_rect.height()*h);
  //cv::RotatedRect rot_rect(center, size, output_rect.rotation());
  //cv::Point2f pts[4];
  // rot_rect.points(pts);
  // for (int i = 0; i < 4; i++){
  //   cv::line(myInputImg, pts[i], pts[(i+1)%4], cv::Scalar(0,255,255),4);
  // }
  return graph.WaitUntilDone();
}     

float thinPlateSpline(const Point2f& p1, const Point2f& p2) {
  auto d = sqrt(pow(p1.x-p2.x, 2) + pow(p1.y-p2.y, 2));
  if (d == 0) return 1;
  return d*d*log(d);
}

std::tuple<Mat,Mat> getRBFWeight(const std::vector<Point2f>& src_landmarks, const std::vector<Point2f>& model_landmarks) {
  Mat del_x(src_landmarks.size(), 1, CV_32F);
  Mat del_y(src_landmarks.size(), 1, CV_32F);
  Mat new_del_x(src_landmarks.size(), 1, CV_32F);
  Mat new_del_y(src_landmarks.size(), 1, CV_32F);
 
  for(int i = 0; i < src_landmarks.size(); i++){
    del_x.at<float>(i) = model_landmarks[i].x - src_landmarks[i].x;
    del_y.at<float>(i) = model_landmarks[i].y - src_landmarks[i].y;
    new_del_x.at<float>(i) = del_x.at<float>(i) - model_landmarks[i].x;
    new_del_y.at<float>(i) = del_y.at<float>(i) - model_landmarks[i].y;
    
  }
  
  Mat matrix(src_landmarks.size(),src_landmarks.size(), CV_32F);
  for(int i = 0; i < matrix.rows; i++){
    for (int j = 0; j < matrix.cols; j++){
      matrix.at<float>(i,j) = thinPlateSpline(src_landmarks[j], src_landmarks[i]);
    }
  }
  
  Mat weight_x(src_landmarks.size(),1,CV_32F);
  Mat weight_y(src_landmarks.size(),1,CV_32F);
  solve(matrix, new_del_x, weight_x);  
  solve(matrix, new_del_y, weight_y);

  return std::make_tuple(weight_x, weight_y);
}

std::tuple<Mat,Mat> RBF(const std::vector<Point2f>& src_landmarks, const std::vector<Point2f>& model_landmarks, const Mat& srcImg, const std::tuple<Mat,Mat>& weight){
  
  Mat del_x(src_landmarks.size(), 1, CV_32F);
  Mat del_y(src_landmarks.size(), 1, CV_32F);
  Mat samplePoints_x(src_landmarks.size(), 2, CV_32F);
  Mat samplePoints_y(src_landmarks.size(), 2, CV_32F);
  for (int i = 0 ; i < src_landmarks.size(); i++){
    del_x.at<float>(i) = model_landmarks[i].x - src_landmarks[i].x;
    del_y.at<float>(i) = model_landmarks[i].y - src_landmarks[i].y;
    samplePoints_x.at<float>(i,0) = src_landmarks[i].x;
    samplePoints_x.at<float>(i,1) = (float)1;
    samplePoints_y.at<float>(i,0) = src_landmarks[i].y;
    samplePoints_y.at<float>(i,1) = (float)1;
  }
  Mat coeff_x(2, 1, CV_32F);
  Mat coeff_y(2, 1, CV_32F);
  solve(samplePoints_x, del_x, coeff_x, DECOMP_SVD);
  solve(samplePoints_y, del_y, coeff_y, DECOMP_SVD);
  std::cout << coeff_x << std::endl;
  std::cout << coeff_y << std::endl;
  Mat map_x = Mat::zeros(srcImg.size(), CV_32F);
  Mat map_y = Mat::zeros(srcImg.size(), CV_32F);
  for (int row = 0; row < srcImg.rows; row++){
    for (int col = 0; col < srcImg.cols; col++){
      for (int k = 0; k < src_landmarks.size(); k++){
        map_x.at<float>(row,col) -= std::get<0>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(row,col)); 
        map_y.at<float>(row,col) -= std::get<1>(weight).at<float>(k)*thinPlateSpline(src_landmarks[k], Point2f(col,row));
      }
      map_x.at<float>(row,col) += col*coeff_x.at<float>(0,0) + coeff_x.at<float>(1,0);
      map_y.at<float>(row,col) += row*coeff_y.at<float>(0,0) + coeff_y.at<float>(1,0);
      // map_x.at<float>(row,col) += col;
      // map_y.at<float>(row,col) += row;
    }
  }
  return std::make_tuple(map_x, map_y);

  // 잘 나온 것
  // Mat point_x = Mat::zeros(src_landmarks.size(),1, CV_32F);
  // Mat point_y = Mat::zeros(src_landmarks.size(),1, CV_32F);
  // for (int i = 0; i < src_landmarks.size(); i++){
  //   for (int j = 0; j < src_landmarks.size(); j++){
  //     point_x.at<float>(i,0) += std::get<0>(weight).at<float>(j)*thinPlateSpline(src_landmarks[j], src_landmarks[i]); 
  //     point_y.at<float>(i,0) += std::get<1>(weight).at<float>(j)*thinPlateSpline(src_landmarks[j], src_landmarks[i]);
  //   }
  //   point_x.at<float>(i,0) += src_landmarks[i].x;
  //   point_y.at<float>(i,0) += src_landmarks[i].y;
  // }
  
  //  std::cout<< "weight delta_x" << point_x << std::endl;
  //  std::cout<< "weight delta_y" << point_y << std::endl;
  //  return std::make_tuple(point_x, point_y);
}


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  Mat myInputImg = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/ms.jpg");
  Mat myInputImg2 = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/jehoon.jpg");
  mediapipe::NormalizedLandmarkList landmarks;
  mediapipe::NormalizedLandmarkList landmarks2;
  RunMPPGraph(myInputImg, landmarks);
  RunMPPGraph(myInputImg2, landmarks2);
  std::vector<Point2f> p1;
  std::vector<Point2f> p2;
  
  for (int i = 0; i < landmarks.landmark_size(); i++){
    p1.push_back(Point2f((float)landmarks.landmark(i).x()*myInputImg.cols,(float)landmarks.landmark(i).y()*myInputImg.rows));
    p2.push_back(Point2f((float)landmarks2.landmark(i).x()*myInputImg2.cols,(float)landmarks2.landmark(i).y()*myInputImg2.rows));
    // 첫 번째 이미지에 랜드마크 빨간 점으로 표시
    //circle(myInputImg, Point(landmarks.landmark(i).x()*myInputImg.cols, landmarks.landmark(i).y()*myInputImg.rows), 1, Scalar(0, 0, 255), - 1);
    // 두 번째 이미지에 랜드마크 파란 점으로 표시
    circle(myInputImg2, Point(landmarks2.landmark(i).x()*myInputImg2.cols, landmarks2.landmark(i).y()*myInputImg2.rows), 1, Scalar(255, 0, 0), - 1);
  }
  //std::cout<<p2<<std::endl;

  Mat H = findHomography(p1, p2);
  Mat imgwarp;
  warpPerspective(myInputImg, imgwarp, H, Size(myInputImg.cols*1.5 , myInputImg.rows *1.5));
  std::vector<Point2f> pointwarp;
  perspectiveTransform(p1, pointwarp, H); 
  for (int i = 0; i < landmarks.landmark_size(); i++){
    // 두 번째 이미지에 첫번째 랜드마크 빨간 점으로 표시
    circle(myInputImg2, pointwarp[i], 1, Scalar(0, 0, 225), - 1);
  }

  auto weight = getRBFWeight(pointwarp, p2);
  Mat dst(myInputImg.size(), myInputImg.type());
  
  auto [map_x, map_y] = RBF(p1, p2, imgwarp, weight);
  remap(imgwarp, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
  //imshow("img0", myInputImg);
  imshow("img1", imgwarp);
  imshow("img2", myInputImg2);
  imshow("img3", dst);
  waitKey();
  
  return EXIT_SUCCESS;
}