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

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "mediapipe/calculators/util/landmarks_to_render_data_calculator.h"


constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";
//constexpr char kMultiLandmarksStream[] = "multi_face_landmarks";
constexpr char kLandmarksStream[] = "single_face_landmarks";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

absl::Status RunMPPGraph() {
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
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                      graph.AddOutputStreamPoller(kOutputStream));
  //MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_multi_landmark,graph.AddOutputStreamPoller(kMultiLandmarksStream));
  MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_single_landmark,
                      graph.AddOutputStreamPoller(kLandmarksStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  cv::Mat myInputImg;
  myInputImg = cv::imread("C:/Users/yeon/mediapipe_repo/mediapipe/mediapipe/examples/desktop/onew3.jpg");
  cv::Mat inputImg;
  cv::cvtColor(myInputImg, inputImg, cv::COLOR_BGR2RGB);  

  auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGB, inputImg.cols, inputImg.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  inputImg.copyTo(input_frame_mat);

  // Send image packet into the graph.
  size_t frame_timestamp_us =
      (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
  MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
      kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

  // Get the graph result packet, or stop if that fails.
  mediapipe::Packet packet;
  mediapipe::Packet multi_landmark_packet;
  mediapipe::Packet landmark_packet;
  if (!poller.Next(&packet)) ABSL_LOG(INFO) << "stop graph";
  //if (!poller_multi_landmark.Next(&multi_landmark_packet)) ABSL_LOG(INFO) << "stop graph";
  if (!poller_single_landmark.Next(&landmark_packet)) ABSL_LOG(INFO) << "stop graph";
  //ABSL_LOG(INFO) << landmark_packet.DebugTypeName();
  auto& output_frame = packet.Get<mediapipe::ImageFrame>();
  auto& output_landmarks = landmark_packet.Get<mediapipe::NormalizedLandmarkList>();
  //auto& output_multi_landmarks = multi_landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
  //std::cout << "size:" << output_multi_landmarks.size() << std::endl;
  /*
  for (const mediapipe::NormalizedLandmarkList &normalizedlandmarkList : output_multi_landmarks)
    {
      std::cout << "FaceLandmarks:";
      std::cout << normalizedlandmarkList.DebugString();
      std::cout << std::cout << output_landmarks[0].NormalizedLandmark;
    }
  */
 
  for (int i = 0; i < output_landmarks.landmark_size(); i++){
        std::cout<< i+1 << "th landmark:" << std::endl;
        std::cout<< output_landmarks.landmark(i).x() <<std::endl;
        std::cout<< output_landmarks.landmark(i).y() <<std::endl;
        std::cout<< output_landmarks.landmark(i).z() <<std::endl;
    }
  float w = 600;
  float h = 600;
  cv::Mat draw_mat(w, h, CV_8UC3);
  for (int i = 0; i < output_landmarks.landmark_size(); i++){
    cv::circle(draw_mat, cv::Point(output_landmarks.landmark(i).x()*w, output_landmarks.landmark(i).y()*h), 1, cv::Scalar(0, 0, 255), - 1);
  }
  //cv::circle(draw_mat, cv::Point(100, 300), 10, cv::Scalar(0, 255, 255), - 1);
  imshow("img", draw_mat);

  // Convert back to opencv for display or saving.
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
  //cv::imshow(kWindowName, output_frame_mat);
  cv::waitKey();
  ABSL_LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}
      

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}