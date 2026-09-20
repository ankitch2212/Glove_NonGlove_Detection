// utils.hpp — small free functions shared across the app: file listing,
// class-name loading, greedy NMS, drawing, and dependency-free JSON writing.
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "detector.hpp"

// Reads classes.txt, one class name per line (blank lines skipped).
std::vector<std::string> load_class_names(const std::string& path);

// Lists image files (.jpg/.jpeg/.png, case-insensitive) in a directory,
// sorted by filename for reproducible ordering.
std::vector<std::filesystem::path> list_images(const std::string& dir);

// Intersection-over-union of two axis-aligned rectangles.
float iou(const cv::Rect& a, const cv::Rect& b);

// Greedy, class-aware non-maximum suppression. Returns the indices (into the
// input vectors) of the boxes to keep, highest score first.
//
// The exported ONNX graph has no built-in NMS (nms=False at export time), so
// this is a from-scratch implementation: sort all candidate boxes by score,
// walk them best-first, and suppress any later box that overlaps a kept box
// of the SAME class beyond iou_threshold. Class-aware means a gloved_hand box
// never suppresses an overlapping bare_hand box (e.g. the same hand region
// scored ambiguously) -- only same-class duplicates are merged.
std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                      const std::vector<float>& scores,
                      const std::vector<int>& class_ids,
                      float iou_threshold);

// Draws all detections onto img in place: a colored box plus a filled label
// chip (so text stays legible on any background), clamped inside the frame.
void draw_detections(cv::Mat& img, const std::vector<Detection>& dets);

// Escapes a string for embedding in a JSON string literal (minimal: quotes,
// backslashes, and control characters -- sufficient for filenames/labels).
std::string json_escape(const std::string& s);

// Writes the aggregate detections.json: a JSON array of
// {"filename": ..., "detections": [{"label":...,"confidence":...,"bbox":[...]}]}
// objects, one per processed image, in the same schema as Part 1's per-image
// logs. Confidence is formatted to 2 decimal places to match the spec example.
void write_json_log(const std::string& path,
                     const std::vector<std::pair<std::string, std::vector<Detection>>>& all_detections);
