#include "detector.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "utils.hpp"

Detector::Detector(const Config& cfg)
    : cfg_(cfg),
      env_(ORT_LOGGING_LEVEL_WARNING, "glove_detector"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    class_names_ = load_class_names(cfg_.classes_path);

    session_options_.SetIntraOpNumThreads(cfg_.num_threads);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
    std::wstring wpath(cfg_.model_path.begin(), cfg_.model_path.end());
    session_ = std::make_unique<Ort::Session>(env_, wpath.c_str(), session_options_);
#else
    session_ = std::make_unique<Ort::Session>(env_, cfg_.model_path.c_str(), session_options_);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    input_name_ = session_->GetInputNameAllocated(0, allocator).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator).get();

    // Sanity-check the output tensor shape against the number of classes we
    // expect. The exported graph is [1, 4 + num_classes, num_anchors] --
    // channel-major, NOT [1, num_anchors, 4 + num_classes]. Getting this
    // assertion right up front is what makes a shape mismatch (e.g. a model
    // exported at a different imgsz, or with a different class count) fail
    // loudly at startup instead of producing silently-garbage detections.
    auto output_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 3 ||
        output_shape[1] != static_cast<int64_t>(4 + class_names_.size())) {
        std::ostringstream msg;
        msg << "model output shape mismatch: expected [1, " << (4 + class_names_.size())
            << ", N] (4 box coords + " << class_names_.size() << " classes), got [";
        for (size_t i = 0; i < output_shape.size(); ++i) msg << output_shape[i] << (i + 1 < output_shape.size() ? ", " : "");
        msg << "]. Check that classes.txt matches the exported model, and that the model "
               "was exported with nms=False.";
        throw std::runtime_error(msg.str());
    }

    std::cerr << "[detector] loaded '" << cfg_.model_path << "' with " << class_names_.size()
              << " classes, input " << cfg_.input_width << "x" << cfg_.input_height << "\n";
}

cv::Mat Detector::preprocess(const cv::Mat& bgr, LetterboxInfo& info) const {
    int target_w = cfg_.input_width;
    int target_h = cfg_.input_height;

    float scale = std::min(static_cast<float>(target_w) / bgr.cols,
                            static_cast<float>(target_h) / bgr.rows);
    int new_w = static_cast<int>(std::round(bgr.cols * scale));
    int new_h = static_cast<int>(std::round(bgr.rows * scale));

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    int pad_x = (target_w - new_w) / 2;
    int pad_y = (target_h - new_h) / 2;
    int pad_right = target_w - new_w - pad_x;
    int pad_bottom = target_h - new_h - pad_y;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_y, pad_bottom, pad_x, pad_right,
                        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    info.scale = scale;
    info.pad_x = pad_x;
    info.pad_y = pad_y;

    // blobFromImage handles BGR->RGB swap, 1/255 scaling, and HWC->NCHW in one call.
    cv::Mat blob;
    cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), /*swapRB=*/true,
                            /*crop=*/false, CV_32F);
    return blob;
}

std::vector<Detection> Detector::postprocess(const float* data,
                                              int64_t num_anchors,
                                              const LetterboxInfo& info,
                                              const cv::Size& original_size) const {
    const int64_t nc = static_cast<int64_t>(class_names_.size());
    const int64_t na = num_anchors;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(na);
    scores.reserve(na);
    class_ids.reserve(na);

    // Output layout is channel-major: [1, 4+nc, na] -- i.e. all `na` cx
    // values first, then all `na` cy values, etc. It is NOT [1, na, 4+nc].
    // YOLOv8/v11 (unlike v5) has no separate objectness channel, and the
    // per-class scores are already sigmoid-activated by the exported graph,
    // so we take the best class score directly with no further activation.
    for (int64_t i = 0; i < na; ++i) {
        float cx = data[0 * na + i];
        float cy = data[1 * na + i];
        float w = data[2 * na + i];
        float h = data[3 * na + i];

        int best_class = -1;
        float best_score = 0.0f;
        for (int64_t c = 0; c < nc; ++c) {
            float s = data[(4 + c) * na + i];
            if (s > best_score) {
                best_score = s;
                best_class = static_cast<int>(c);
            }
        }
        if (best_class < 0 || best_score < cfg_.confidence_threshold) continue;

        // Un-letterbox: model-space (416x416, padded) -> original image pixels.
        float x1 = (cx - w * 0.5f - info.pad_x) / info.scale;
        float y1 = (cy - h * 0.5f - info.pad_y) / info.scale;
        float x2 = (cx + w * 0.5f - info.pad_x) / info.scale;
        float y2 = (cy + h * 0.5f - info.pad_y) / info.scale;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(original_size.width - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(original_size.height - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(original_size.width - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(original_size.height - 1));
        if (x2 <= x1 || y2 <= y1) continue;

        boxes.emplace_back(cv::Point(static_cast<int>(x1), static_cast<int>(y1)),
                            cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    std::vector<int> keep = nms(boxes, scores, class_ids, cfg_.nms_threshold);

    std::vector<Detection> results;
    results.reserve(keep.size());
    for (int idx : keep) {
        results.push_back(Detection{class_ids[idx], class_names_[class_ids[idx]], scores[idx], boxes[idx]});
    }
    return results;
}

std::vector<Detection> Detector::detect(const cv::Mat& bgr) const {
    LetterboxInfo info;
    cv::Mat blob = preprocess(bgr, info);

    std::array<int64_t, 4> input_shape = {1, 3, cfg_.input_height, cfg_.input_width};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, reinterpret_cast<float*>(blob.data), blob.total(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
                                  output_names, 1);

    const float* out_data = outputs[0].GetTensorData<float>();
    auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, 4+nc, na]
    int64_t num_anchors = out_shape[2];

    return postprocess(out_data, num_anchors, info, bgr.size());
}
