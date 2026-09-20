// config.hpp — plain struct + a minimal hand-rolled YAML-subset loader.
//
// We deliberately do NOT vendor a YAML library. config.yaml is a flat
// "key: value" map with no nesting, lists, or anchors -- a ~60 line parser
// that handles exactly that (and documents its own limits) is a smaller,
// more honest dependency footprint than pulling in yaml-cpp or a header-only
// library just to read six scalar fields. See load_config() in config.cpp
// for the parsing rules.
#pragma once

#include <string>

struct Config {
    // These defaults are written relative to config.yaml's OWN directory
    // (config/), matching how main.cpp resolves every path -- so the binary
    // finds the right files even if config.yaml is missing or incomplete.
    std::string model_path        = "../models/glove_detector.onnx";
    std::string classes_path      = "../models/classes.txt";
    std::string input_dir         = "../input";
    std::string output_dir        = "../output";
    std::string log_path          = "../logs/detections.json";
    float       confidence_threshold = 0.5f;
    float       nms_threshold        = 0.45f;
    int         input_width       = 416;
    int         input_height      = 416;
    int         num_threads       = 4;
};

// Loads a flat "key: value" YAML file into a Config, starting from defaults.
// - Missing file: prints a warning to stderr and returns the defaults untouched
//   (the program still runs correctly with no config.yaml present).
// - Missing keys: left at their struct default.
// - Unknown keys: warning to stderr, parsing continues (forward-compatible).
// - Malformed numeric values: warning to stderr, that field keeps its default.
Config load_config(const std::string& path);

// Resolves a path from the config relative to config.yaml's own directory,
// so the binary finds models/input/output/log paths correctly no matter what
// directory it was launched from (e.g. running from build/Release/).
std::string resolve_relative(const std::string& base_dir, const std::string& maybe_relative);
