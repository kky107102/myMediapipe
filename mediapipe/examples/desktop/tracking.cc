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

absl::Status RunMPPGraph(Mat img, mediapipe::NormalizedRect& output_rect) {
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
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_rect, graph.AddOutputStreamPoller(kROIStream));
  //MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_single_landmark, graph.AddOutputStreamPoller(kLandmarksStream));
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
  //mediapipe::Packet landmark_packet;
  mediapipe::Packet rect_packet;
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  //if (!poller_single_landmark.Next(&landmark_packet)) ABSL_LOG(INFO) << "stop graph";
  if (!poller_rect.Next(&rect_packet)) ABSL_LOG(INFO) << "stop graph";
  //output_landmarks = landmark_packet.Get<mediapipe::NormalizedLandmarkList>();
  output_rect = rect_packet.Get<mediapipe::NormalizedRect>();
  return graph.WaitUntilDone();
}    


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);

  Mat myInputImg = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/jehoon.jpg");
  //Mat myInputImg2 = imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/jehoon.jpg");
  
  mediapipe::NormalizedRect rect;
  //mediapipe::NormalizedRect rect2;
  RunMPPGraph(myInputImg, rect);
  //RunMPPGraph(myInputImg2, rect2);
  cv::Point2f center(rect.x_center()*myInputImg.rows, rect.y_center()*myInputImg.cols);
  //cv::Point2f center(rect2.x_center()*w, rect2.y_center()*h);

  std::cout << rect.rotation();
  cv::Size2f size(rect.width()*myInputImg.rows, rect.height()*myInputImg.cols);
  cv::RotatedRect rot_rect(center, size, rect.rotation());
  cv::Point2f pts[4];
  rot_rect.points(pts);
  for (int i = 0; i < 4; i++){
    cv::line(myInputImg, pts[i], pts[(i+1)%4], cv::Scalar(0,255,255),4);
  }
  
  imshow("img1", myInputImg);
  waitKey();
  
  return EXIT_SUCCESS;
}