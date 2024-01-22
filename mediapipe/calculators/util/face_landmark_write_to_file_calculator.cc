#include <string>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/canonical_errors.h"
#include "mediapipe/tasks/cc/components/containers/landmark.h"

namespace mediapipe {

class FaceLandmarkWriteToFileCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("LANDMARKS").Set<NormalizedLandmarkList>();
    return absl::OkStatus();
    }

  absl::Status Open(CalculatorContext* cc) final { return absl::OkStatus(); }

  absl::Status Process(CalculatorContext* cc) final {
    const auto& input_landmarks =
    cc->Inputs().Tag("LANDMARKS").Get<NormalizedLandmarkList>();
    /*    
    std::string label;
    label = input_handedness.classification(0).label();
    if (label.compare("Right") == 1) {
        std::cout << "Right: " << input_landmarks.landmark(0).x() << std::endl;
    } else {
        std::cout << "Left : " << input_landmarks.landmark(0).x() << std::endl;
    }
    */
    for (int i = 0; i < input_landmarks.landmark_size(); i++){
        std::cout<< i+1 << "번째 landmark:" << std::endl;
        std::cout<< input_landmarks.landmark(i).x() <<std::endl;
        std::cout<< input_landmarks.landmark(i).y() <<std::endl;
        std::cout<< input_landmarks.landmark(i).z() <<std::endl;
    }
    return absl::OkStatus();
    }

};
REGISTER_CALCULATOR(FaceLandmarkWriteToFileCalculator);

}  // namespace mediapipe