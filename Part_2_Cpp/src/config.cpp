#include "config.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Strip a trailing "# comment" -- only when the '#' is preceded by whitespace
// or starts the line, so a value like "color: #114" (not used here, but a
// reasonable YAML-ism) would survive. Our config has no such values today;
// this is a defensive rule, not a used feature.
std::string strip_comment(const std::string& line) {
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '#' && (i == 0 || std::isspace(static_cast<unsigned char>(line[i - 1])))) {
            return line.substr(0, i);
        }
    }
    return line;
}

std::string strip_quotes(std::string s) {
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

bool try_parse_float(const std::string& s, float& out) {
    try {
        size_t pos = 0;
        out = std::stof(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool try_parse_int(const std::string& s, int& out) {
    try {
        size_t pos = 0;
        out = std::stoi(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

}  // namespace

Config load_config(const std::string& path) {
    Config cfg;

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[config] warning: could not open '" << path
                  << "' -- using built-in defaults\n";
        return cfg;
    }

    std::string line;
    int line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        line = strip_comment(line);
        line = trim(line);
        if (line.empty()) continue;

        size_t colon = line.find(':');
        if (colon == std::string::npos) {
            std::cerr << "[config] warning: " << path << ":" << line_no
                      << " has no ':' -- skipping: \"" << line << "\"\n";
            continue;
        }

        std::string key = trim(line.substr(0, colon));
        std::string value = strip_quotes(trim(line.substr(colon + 1)));
        if (value.empty()) {
            std::cerr << "[config] warning: " << path << ":" << line_no
                      << " key '" << key << "' has an empty value -- keeping default\n";
            continue;
        }

        if (key == "model_path") {
            cfg.model_path = value;
        } else if (key == "classes_path") {
            cfg.classes_path = value;
        } else if (key == "input_dir") {
            cfg.input_dir = value;
        } else if (key == "output_dir") {
            cfg.output_dir = value;
        } else if (key == "log_path") {
            cfg.log_path = value;
        } else if (key == "confidence_threshold") {
            float v;
            if (try_parse_float(value, v)) cfg.confidence_threshold = v;
            else std::cerr << "[config] warning: bad float for confidence_threshold: \"" << value << "\"\n";
        } else if (key == "nms_threshold") {
            float v;
            if (try_parse_float(value, v)) cfg.nms_threshold = v;
            else std::cerr << "[config] warning: bad float for nms_threshold: \"" << value << "\"\n";
        } else if (key == "input_width") {
            int v;
            if (try_parse_int(value, v)) cfg.input_width = v;
            else std::cerr << "[config] warning: bad int for input_width: \"" << value << "\"\n";
        } else if (key == "input_height") {
            int v;
            if (try_parse_int(value, v)) cfg.input_height = v;
            else std::cerr << "[config] warning: bad int for input_height: \"" << value << "\"\n";
        } else if (key == "num_threads") {
            int v;
            if (try_parse_int(value, v)) cfg.num_threads = v;
            else std::cerr << "[config] warning: bad int for num_threads: \"" << value << "\"\n";
        } else {
            std::cerr << "[config] warning: " << path << ":" << line_no
                      << " unknown key '" << key << "' -- ignored\n";
        }
    }

    return cfg;
}

std::string resolve_relative(const std::string& base_dir, const std::string& maybe_relative) {
    fs::path p(maybe_relative);
    if (p.is_absolute()) return maybe_relative;
    return (fs::path(base_dir) / p).lexically_normal().string();
}
