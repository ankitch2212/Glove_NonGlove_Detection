// main.cpp — CLI entry point for the C++ glove/bare-hand detector.
//
// Usage:
//   glove_detector.exe [--config path/to/config.yaml]
//                       [--input DIR] [--output DIR] [--confidence 0.5]
//
// The three optional overrides mirror Part 1's Python CLI surface. Without
// any of them, everything comes from config.yaml (or its built-in defaults
// if the file is missing).
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "detector.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

namespace {

struct CliArgs {
    std::string config_path = "config/config.yaml";
    std::optional<std::string> input_dir;
    std::optional<std::string> output_dir;
    std::optional<float> confidence;
};

CliArgs parse_args(int argc, char** argv) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (arg == "--config") args.config_path = next("--config");
        else if (arg == "--input") args.input_dir = next("--input");
        else if (arg == "--output") args.output_dir = next("--output");
        else if (arg == "--confidence") args.confidence = std::stof(next("--confidence"));
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: glove_detector [--config path] [--input dir] [--output dir] "
                         "[--confidence 0.5]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CliArgs args = parse_args(argc, argv);

        Config cfg = load_config(args.config_path);

        // Resolve every path in the config relative to config.yaml's own
        // directory, so the binary works whether it's launched from the
        // project root or from build/Release/.
        std::string base_dir = fs::path(args.config_path).has_parent_path()
                                    ? fs::path(args.config_path).parent_path().string()
                                    : ".";
        cfg.model_path = resolve_relative(base_dir, cfg.model_path);
        cfg.classes_path = resolve_relative(base_dir, cfg.classes_path);
        cfg.input_dir = resolve_relative(base_dir, cfg.input_dir);
        cfg.output_dir = resolve_relative(base_dir, cfg.output_dir);
        cfg.log_path = resolve_relative(base_dir, cfg.log_path);

        // CLI overrides win over the config file.
        if (args.input_dir) cfg.input_dir = *args.input_dir;
        if (args.output_dir) cfg.output_dir = *args.output_dir;
        if (args.confidence) cfg.confidence_threshold = *args.confidence;

        fs::create_directories(cfg.output_dir);
        fs::create_directories(fs::path(cfg.log_path).parent_path());

        Detector detector(cfg);

        auto images = list_images(cfg.input_dir);
        if (images.empty()) {
            std::cerr << "error: no .jpg/.jpeg/.png images found in '" << cfg.input_dir << "'\n";
            return 1;
        }

        std::vector<std::pair<std::string, std::vector<Detection>>> all_detections;
        int total_detections = 0, gloved = 0, bare = 0;

        auto t_start = std::chrono::steady_clock::now();
        for (const auto& img_path : images) {
            cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
            if (img.empty()) {
                std::cerr << "warning: could not read '" << img_path.string() << "' -- skipping\n";
                continue;
            }

            auto t0 = std::chrono::steady_clock::now();
            std::vector<Detection> dets = detector.detect(img);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            draw_detections(img, dets);
            fs::path out_path = fs::path(cfg.output_dir) / img_path.filename();
            cv::imwrite(out_path.string(), img);

            for (const auto& d : dets) {
                if (d.class_id == 0) ++gloved; else ++bare;
            }
            total_detections += static_cast<int>(dets.size());

            std::cout << img_path.filename().string() << ": " << dets.size()
                      << " detection(s) in " << ms << " ms\n";

            all_detections.emplace_back(img_path.filename().string(), std::move(dets));
        }
        auto t_end = std::chrono::steady_clock::now();
        double total_s = std::chrono::duration<double>(t_end - t_start).count();

        write_json_log(cfg.log_path, all_detections);

        std::cout << "\n" << images.size() << " images processed, " << total_detections
                  << " detections (" << gloved << " gloved_hand, " << bare << " bare_hand), "
                  << total_s << "s elapsed\n";
        std::cout << "Annotated images -> " << cfg.output_dir << "\n";
        std::cout << "Detection log     -> " << cfg.log_path << "\n";

        return 0;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
