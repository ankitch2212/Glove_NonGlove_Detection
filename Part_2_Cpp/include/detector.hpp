// detector.hpp — ONNX Runtime-backed YOLO detector for gloved_hand / bare_hand.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "config.hpp"

struct Detection {
    int         class_id;
    std::string label;
    float       confidence;
    cv::Rect    box;   // xyxy-derived rect, in ORIGINAL image pixel coordinates
};

// Scale + padding applied by letterbox(), needed to map model-space boxes
// (416x416) back to the coordinates of the original input image.
struct LetterboxInfo {
    float scale = 1.0f;
    int   pad_x = 0;
    int   pad_y = 0;
};

class Detector {
public:
    explicit Detector(const Config& cfg);

    // Runs the full pipeline on one BGR image: letterbox -> inference ->
    // decode the [1, 4+nc, num_anchors] output -> un-letterbox -> NMS.
    std::vector<Detection> detect(const cv::Mat& bgr) const;

    const std::vector<std::string>& classes() const noexcept { return class_names_; }

private:
    cv::Mat preprocess(const cv::Mat& bgr, LetterboxInfo& info) const;

    std::vector<Detection> postprocess(const float* data,
                                        int64_t num_anchors,
                                        const LetterboxInfo& info,
                                        const cv::Size& original_size) const;

    Config                          cfg_;
    std::vector<std::string>        class_names_;

    Ort::Env                        env_;
    Ort::SessionOptions             session_options_;
    std::unique_ptr<Ort::Session>   session_;
    Ort::MemoryInfo                 memory_info_;

    std::string input_name_;
    std::string output_name_;
};
