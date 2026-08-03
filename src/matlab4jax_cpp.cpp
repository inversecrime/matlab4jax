#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>

#include "nanobind/nanobind.h"
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"

namespace nb = nanobind;

typedef xla::ffi::Span<const uint8_t> StringAsBytes;
typedef xla::ffi::Span<const uint8_t> StringListAsBytes;

static std::mutex mutex;

static const std::string function_name = "matlab4jax_temp";
static const std::filesystem::path function_dir = []() {
    auto path = std::filesystem::temp_directory_path() / "matlab4jax";
    std::filesystem::create_directories(path);
    return path;
}();
static const std::filesystem::path function_file = function_dir / (function_name + ".m");

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vector) {
    out << "[";
    for (size_t i = 0; i < vector.size(); i++) {
        out << vector[i];
        if (i != vector.size() - 1) {
            out << ", ";
        }
    }
    out << "]";
    return out;
}

std::string join(const std::string &separator, const std::vector<std::string> &strings) {
    std::string result;
    for (size_t i = 0; i < strings.size(); i++) {
        result += strings[i];
        if (i + 1 < strings.size()) {
            result += separator;
        }
    }
    return result;
}

std::u16string string_to_utf16(const std::string &string) {
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    return convert.from_bytes(string);
}

std::string decode_string(const StringAsBytes &string_as_bytes) {
    return std::string(string_as_bytes.begin(), string_as_bytes.end());
}

std::vector<std::string> decode_string_list(const StringListAsBytes &string_list_as_bytes) {
    std::vector<std::string> string_list;
    size_t i = 0;

    while (i < string_list_as_bytes.size()) {
        if (i + 4 > string_list_as_bytes.size()) {
            throw std::runtime_error("unexpected end of span");
        }
        uint32_t string_length = *reinterpret_cast<const uint32_t *>(&string_list_as_bytes[i]);
        i += 4;

        if (i + string_length > string_list_as_bytes.size()) {
            throw std::runtime_error("unexpected end of span");
        }
        std::string string_content = std::string(reinterpret_cast<const char *>(&string_list_as_bytes[i]), string_length);
        i += string_length;

        string_list.push_back(string_content);
    }

    return string_list;
}

matlab::engine::MATLABEngine &get_engine() {
    static auto engine = []() {
        auto engine = matlab::engine::startMATLAB();
        engine->eval(string_to_utf16("addpath('" + function_dir.string() + "');"));
        return engine;
    }();
    return *engine;
}

template <typename T>
std::vector<size_t> get_shape(const nb::ndarray<nb::jax, T, nb::c_contig> &jax_array) {
    std::vector<size_t> shape;
    for (size_t i = 0; i < jax_array.ndim(); i++) {
        shape.push_back(jax_array.shape(i));
    }
    if (shape.size() == 0) {
        shape.push_back(1);
        shape.push_back(1);
    }
    if (shape.size() == 1) {
        shape.insert(shape.begin(), 1);
    }
    return shape;
}

template <xla::ffi::DataType T>
std::vector<size_t> get_shape(const xla::ffi::Buffer<T> &xla_array) {
    std::vector<size_t> shape;
    for (int64_t x : xla_array.dimensions()) {
        shape.push_back(x);
    }
    if (shape.size() == 0) {
        shape.push_back(1);
        shape.push_back(1);
    }
    if (shape.size() == 1) {
        shape.insert(shape.begin(), 1);
    }
    return shape;
}

template <typename T>
matlab::data::TypedArray<T> jax_to_matlab(const nb::ndarray<nb::jax, T, nb::c_contig> &jax_array) {
    matlab::data::ArrayFactory factory;
    std::vector<size_t> shape = get_shape(jax_array);
    return factory.createArray(shape, jax_array.data(), jax_array.data() + jax_array.size(), matlab::data::InputLayout::ROW_MAJOR);
}

template <xla::ffi::DataType T>
matlab::data::TypedArray<xla::ffi::NativeType<T>> xla_to_matlab(const xla::ffi::Buffer<T> &xla_array) {
    matlab::data::ArrayFactory factory;
    std::vector<size_t> shape = get_shape(xla_array);
    return factory.createArray(shape, xla_array.typed_data(), xla_array.typed_data() + xla_array.element_count());
}

template <typename T>
struct owner {
    std::vector<std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>> data;
    std::vector<size_t> shape;
};

template <typename T>
nb::ndarray<nb::jax, T, nb::f_contig> matlab_to_jax(const matlab::data::TypedArray<T> &matlab_array) {
    auto *o = new owner<T>{
        std::vector<std::conditional_t<std::is_same_v<T, bool>, uint8_t, T>>(matlab_array.cbegin(), matlab_array.cend()),
        std::vector<size_t>(matlab_array.getDimensions())
    };
    nb::capsule capsule(o, [](void *p) noexcept {
        delete static_cast<owner<T> *>(p);
    });
    return nb::ndarray<nb::jax, T, nb::f_contig>(o->data.data(), o->shape.size(), o->shape.data(), capsule);
}

template <xla::ffi::DataType T>
void matlab_to_xla(const matlab::data::TypedArray<xla::ffi::NativeType<T>> &matlab_array, xla::ffi::Result<xla::ffi::Buffer<T>> &xla_array) {
    std::vector<size_t> matlab_shape = matlab_array.getDimensions();
    std::vector<size_t> xla_shape = get_shape(*xla_array);
    xla_shape.resize(matlab_shape.size(), 1);
    if (matlab_shape != xla_shape) {
        std::ostringstream error_message;
        error_message << "dimension error: matlab shape " << matlab_shape << " does not match xla shape " << xla_shape;
        throw std::runtime_error(error_message.str());
    }
    std::copy(matlab_array.cbegin(), matlab_array.cend(), xla_array->typed_data());
}

void eval_command(const std::string &command) {
    auto &engine = get_engine();
    engine.eval(string_to_utf16(command));
}

void set_variable(const std::string &variable_name, const nb::ndarray<nb::jax, nb::c_contig> &jax_array) {
    auto &engine = get_engine();
    matlab::data::Array matlab_array;
    if (jax_array.dtype() == nb::dtype<std::complex<double>>()) {
        matlab_array = jax_to_matlab<std::complex<double>>(nb::ndarray<nb::jax, std::complex<double>, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<std::complex<float>>()) {
        matlab_array = jax_to_matlab<std::complex<float>>(nb::ndarray<nb::jax, std::complex<float>, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<double>()) {
        matlab_array = jax_to_matlab<double>(nb::ndarray<nb::jax, double, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<float>()) {
        matlab_array = jax_to_matlab<float>(nb::ndarray<nb::jax, float, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<int64_t>()) {
        matlab_array = jax_to_matlab<int64_t>(nb::ndarray<nb::jax, int64_t, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<int32_t>()) {
        matlab_array = jax_to_matlab<int32_t>(nb::ndarray<nb::jax, int32_t, nb::c_contig>(jax_array));
    } else if (jax_array.dtype() == nb::dtype<bool>()) {
        matlab_array = jax_to_matlab<bool>(nb::ndarray<nb::jax, bool, nb::c_contig>(jax_array));
    } else {
        throw std::runtime_error("unsupported jax dtype");
    }
    engine.setVariable(variable_name, std::move(matlab_array));
}

nb::object get_variable(const std::string &variable_name) {
    auto &engine = get_engine();
    matlab::data::Array matlab_array = engine.getVariable(string_to_utf16(variable_name));
    switch (matlab_array.getType()) {
    case matlab::data::ArrayType::COMPLEX_DOUBLE:
        return nb::cast(matlab_to_jax<std::complex<double>>(matlab_array));
    case matlab::data::ArrayType::COMPLEX_SINGLE:
        return nb::cast(matlab_to_jax<std::complex<float>>(matlab_array));
    case matlab::data::ArrayType::DOUBLE:
        return nb::cast(matlab_to_jax<double>(matlab_array));
    case matlab::data::ArrayType::SINGLE:
        return nb::cast(matlab_to_jax<float>(matlab_array));
    case matlab::data::ArrayType::INT64:
        return nb::cast(matlab_to_jax<int64_t>(matlab_array));
    case matlab::data::ArrayType::INT32:
        return nb::cast(matlab_to_jax<int32_t>(matlab_array));
    case matlab::data::ArrayType::LOGICAL:
        return nb::cast(matlab_to_jax<bool>(matlab_array));
    default:
        throw std::runtime_error("unsupported matlab dtype");
    }
}

#define I_TH_ARRAY_TO_ENGINE(T)                                      \
    {                                                                \
        auto variable_name = input_names[i];                         \
        auto xla_array = inputs.get<xla::ffi::Buffer<T>>(i).value(); \
        auto matlab_array = xla_to_matlab(xla_array);                \
        engine.setVariable(variable_name, std::move(matlab_array));  \
    }

#define I_TH_ARRAY_FROM_ENGINE(T)                                               \
    {                                                                           \
        auto variable_name = output_names[i];                                   \
        auto matlab_array = engine.getVariable(string_to_utf16(variable_name)); \
        auto xla_array = outputs.get<xla::ffi::Buffer<T>>(i).value();           \
        matlab_to_xla(matlab_array, xla_array);                                 \
    }

xla::ffi::Error run_matlab_impl(
    StringAsBytes command_as_bytes,
    StringListAsBytes input_names_as_bytes,
    StringListAsBytes output_names_as_bytes,
    bool use_file,
    xla::ffi::RemainingArgs inputs,
    xla::ffi::RemainingRets outputs
) {
    std::lock_guard<std::mutex> lock(mutex);
    auto &engine = get_engine();
    engine.eval(u"clear;");

    std::string command = decode_string(command_as_bytes);
    std::vector<std::string> input_names = decode_string_list(input_names_as_bytes);
    std::vector<std::string> output_names = decode_string_list(output_names_as_bytes);

    for (size_t i = 0; i < input_names.size(); i++) {
        switch (inputs.get<xla::ffi::AnyBuffer>(i).value().element_type()) {
        case xla::ffi::C128:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::C128);
            break;
        case xla::ffi::C64:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::C64);
            break;
        case xla::ffi::F64:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::F64);
            break;
        case xla::ffi::F32:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::F32);
            break;
        case xla::ffi::S64:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::S64);
            break;
        case xla::ffi::S32:
            I_TH_ARRAY_TO_ENGINE(xla::ffi::S32);
            break;
        case xla::ffi::PRED:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::PRED);
            break;
        }
    }

    if (use_file) {
        std::string input_names_joined = join(", ", input_names);
        std::string output_names_joined = join(", ", output_names);
        std::ofstream file(function_file);
        file << "function [" + output_names_joined + "] = " + function_name + "(" + input_names_joined + ")\n";
        file << command + "\n";
        file << "end\n";
        file.close();
        engine.eval(string_to_utf16("[" + output_names_joined + "] = " + function_name + "(" + input_names_joined + ");"));
    } else {
        engine.eval(string_to_utf16(command));
    }

    for (size_t i = 0; i < output_names.size(); i++) {
        switch (outputs.get<xla::ffi::AnyBuffer>(i).value()->element_type()) {
        case xla::ffi::C128:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::C128);
            break;
        case xla::ffi::C64:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::C64);
            break;
        case xla::ffi::F64:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::F64);
            break;
        case xla::ffi::F32:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::F32);
            break;
        case xla::ffi::S64:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::S64);
            break;
        case xla::ffi::S32:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::S32);
            break;
        case xla::ffi::PRED:
            I_TH_ARRAY_FROM_ENGINE(xla::ffi::PRED);
            break;
        };
    }

    engine.eval(u"clear;");
    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    run_matlab,
    run_matlab_impl,
    xla::ffi::Ffi::Bind()
        .Attr<StringAsBytes>("command_as_bytes")
        .Attr<StringListAsBytes>("input_names_as_bytes")
        .Attr<StringListAsBytes>("output_names_as_bytes")
        .Attr<bool>("use_file")
        .RemainingArgs()
        .RemainingRets()
);

NB_MODULE(matlab4jax_cpp, m) {
    m.def("eval_command", &eval_command);
    m.def("set_variable", &set_variable);
    m.def("get_variable", &get_variable);
    m.def("run_matlab", []() { return nb::capsule(reinterpret_cast<void *>(run_matlab)); });
}