#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open classes file: " + path);
    }
    std::string line;
    while (std::getline(file, line)) {
        // strip trailing \r for files with CRLF line endings
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (!line.empty()) names.push_back(line);
    }
    if (names.empty()) {
        throw std::runtime_error("classes file is empty: " + path);
    }
    return names;
}

std::vector<fs::path> list_images(const std::string& dir) {
    static const std::vector<std::string> exts = {".jpg", ".jpeg", ".png"};
    std::vector<fs::path> out;
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        return out;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                        [](unsigned char c) { return std::tolower(c); });
        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
            out.push_back(entry.path());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

float iou(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    if (inter == 0) return 0.0f;
    int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                      const std::vector<float>& scores,
                      const std::vector<int>& class_ids,
                      float iou_threshold) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<int> keep;
    for (size_t i = 0; i < order.size(); ++i) {
        int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int other = order[j];
            if (suppressed[other]) continue;
            if (class_ids[other] != class_ids[idx]) continue;  // class-aware NMS
            if (iou(boxes[idx], boxes[other]) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return keep;
}

void draw_detections(cv::Mat& img, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Scalar color = (d.class_id == 0) ? cv::Scalar(0, 180, 0)   // gloved_hand: green (BGR)
                                              : cv::Scalar(0, 0, 220); // bare_hand:  red   (BGR)
        cv::rectangle(img, d.box, color, 2);

        std::ostringstream text;
        text << d.label << " " << std::fixed << std::setprecision(2) << d.confidence;
        std::string label_text = text.str();

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int chip_x = std::max(0, d.box.x);
        int chip_y = std::max(text_size.height + 4, d.box.y);  // keep the chip inside the frame

        cv::rectangle(img,
                      cv::Point(chip_x, chip_y - text_size.height - 4),
                      cv::Point(chip_x + text_size.width + 4, chip_y),
                      color, cv::FILLED);
        cv::putText(img, label_text, cv::Point(chip_x + 2, chip_y - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

std::string json_escape(const std::string& s) {
    std::ostringstream out;
    for (char c : s) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

void write_json_log(const std::string& path,
                     const std::vector<std::pair<std::string, std::vector<Detection>>>& all_detections) {
    fs::create_directories(fs::path(path).parent_path());

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("could not open log file for writing: " + path);
    }

    out << "[\n";
    for (size_t i = 0; i < all_detections.size(); ++i) {
        const auto& [filename, dets] = all_detections[i];
        out << "  {\n";
        out << "    \"filename\": \"" << json_escape(filename) << "\",\n";
        out << "    \"detections\": [";
        for (size_t j = 0; j < dets.size(); ++j) {
            const auto& d = dets[j];
            out << (j == 0 ? "\n" : ",\n");
            out << "      {\"label\": \"" << json_escape(d.label) << "\", "
                << "\"confidence\": " << std::fixed << std::setprecision(2) << d.confidence << ", "
                << "\"bbox\": [" << d.box.x << ", " << d.box.y << ", "
                << (d.box.x + d.box.width) << ", " << (d.box.y + d.box.height) << "]}";
        }
        out << (dets.empty() ? "]\n" : "\n    ]\n");
        out << "  }" << (i + 1 < all_detections.size() ? "," : "") << "\n";
    }
    out << "]\n";
}
