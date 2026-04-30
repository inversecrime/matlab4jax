#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <iostream>

#include "pybind11/pybind11.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"

namespace py = pybind11;

typedef xla::ffi::Span<const uint8_t> StringAsBytes;
typedef xla::ffi::Span<const uint8_t> StringListAsBytes;

template <typename T>
std::ostream &operator<<(std::ostream &out, std::vector<T> &vector)
{
    out << "[";
    for (size_t i = 0; i < vector.size(); i++)
    {
        out << vector[i];
        if (i != vector.size() - 1)
        {
            out << ", ";
        }
    }
    out << "]";
    return out;
}

std::u16string string_to_utf16(std::string string)
{
    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> convert;
    return convert.from_bytes(string);
}

std::string decode_string(StringAsBytes string_as_bytes)
{
    return std::string(string_as_bytes.begin(), string_as_bytes.end());
}

std::vector<std::string> decode_string_list(StringListAsBytes string_list_as_bytes)
{
    std::vector<std::string> string_list;
    size_t i = 0;

    while (i < string_list_as_bytes.size())
    {
        if (i + 4 > string_list_as_bytes.size())
        {
            throw std::runtime_error("unexpected end of span");
        }
        uint32_t string_length = *reinterpret_cast<const uint32_t *>(&string_list_as_bytes[i]);
        i += 4;

        if (i + string_length > string_list_as_bytes.size())
        {
            throw std::runtime_error("unexpected end of span");
        }
        std::string string_content = std::string(reinterpret_cast<const char *>(&string_list_as_bytes[i]), string_length);
        i += string_length;

        string_list.push_back(string_content);
    }

    return string_list;
}

template <xla::ffi::DataType T>
std::vector<size_t> get_dimensions(xla::ffi::Buffer<T> xla_array)
{
    std::vector<size_t> dimensions;
    for (int64_t x : xla_array.dimensions())
    {
        dimensions.push_back((size_t)(x));
    }
    if (dimensions.size() == 0)
    {
        dimensions.push_back(1);
        dimensions.push_back(1);
    }
    if (dimensions.size() == 1)
    {
        dimensions.insert(dimensions.begin(), 1);
    }
    return dimensions;
}

template <xla::ffi::DataType T>
std::vector<size_t> get_dimensions(xla::ffi::Result<xla::ffi::Buffer<T>> xla_array)
{
    std::vector<size_t> dimensions;
    for (int64_t x : xla_array->dimensions())
    {
        dimensions.push_back((size_t)(x));
    }
    if (dimensions.size() == 0)
    {
        dimensions.push_back(1);
        dimensions.push_back(1);
    }
    if (dimensions.size() == 1)
    {
        dimensions.insert(dimensions.begin(), 1);
    }
    return dimensions;
}

std::vector<size_t> get_dimensions(matlab::data::Array matlab_array)
{
    return matlab_array.getDimensions();
}

template <xla::ffi::DataType T>
matlab::data::TypedArray<xla::ffi::NativeType<T>> xla_to_matlab(xla::ffi::Buffer<T> xla_array)
{
    static matlab::data::ArrayFactory array_factory;

    std::vector<size_t> dimensions = get_dimensions(xla_array);

    return array_factory.createArray(dimensions, xla_array.typed_data(), xla_array.typed_data() + xla_array.element_count());
}

template <xla::ffi::DataType T>
void matlab_to_xla(matlab::data::TypedArray<xla::ffi::NativeType<T>> matlab_array, xla::ffi::Result<xla::ffi::Buffer<T>> xla_array)
{
    std::vector<size_t> matlab_dimensions = get_dimensions(matlab_array);
    std::vector<size_t> xla_dimensions = get_dimensions(xla_array);

    if (matlab_dimensions != xla_dimensions)
    {
        std::ostringstream error_message;
        error_message << "dimension error: matlab dimensions " << matlab_dimensions << " do not match xla dimensions " << xla_dimensions;
        throw std::runtime_error(error_message.str());
    }

    int64_t i = 0;
    for (xla::ffi::NativeType<T> x : matlab_array)
    {
        xla_array->typed_data()[i++] = x;
    }
}

#define I_TH_ARRAY_TO_ENGINE(T)                                                          \
    {                                                                                    \
        std::string variable_name = input_variable_names[i];                             \
        auto xla_variable = inputs.get<xla::ffi::Buffer<T>>(i).value();                  \
        auto matlab_variable = xla_to_matlab(xla_variable);                              \
        engine->setVariable(string_to_utf16(variable_name), std::move(matlab_variable)); \
    }

#define I_TH_ARRAY_FROM_ENGINE(T)                                                   \
    {                                                                               \
        std::string variable_name = output_variable_names[i];                       \
        auto matlab_variable = engine->getVariable(string_to_utf16(variable_name)); \
        auto xla_variable = outputs.get<xla::ffi::Buffer<T>>(i).value();            \
        matlab_to_xla(matlab_variable, xla_variable);                               \
    }

xla::ffi::Error run_matlab_impl(StringAsBytes command_as_bytes,
                                StringListAsBytes input_variable_names_as_bytes,
                                StringListAsBytes output_variable_names_as_bytes,
                                xla::ffi::RemainingArgs inputs,
                                xla::ffi::RemainingRets outputs)
{
    static std::unique_ptr<matlab::engine::MATLABEngine> engine = matlab::engine::startMATLAB();

    std::string command = decode_string(command_as_bytes);
    std::vector<std::string> input_variable_names = decode_string_list(input_variable_names_as_bytes);
    std::vector<std::string> output_variable_names = decode_string_list(output_variable_names_as_bytes);

    for (size_t i = 0; i < input_variable_names.size(); i++)
    {
        switch (inputs.get<xla::ffi::AnyBuffer>(i).value().element_type())
        {
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
        }
    }

    engine->eval(string_to_utf16(command));

    for (size_t i = 0; i < output_variable_names.size(); i++)
    {
        switch (outputs.get<xla::ffi::AnyBuffer>(i).value()->element_type())
        {
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
        };
    }

    return xla::ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(run_matlab, run_matlab_impl,
                              xla::ffi::Ffi::Bind()
                                  .Attr<StringAsBytes>("command_as_bytes")
                                  .Attr<StringListAsBytes>("input_variable_names_as_bytes")
                                  .Attr<StringListAsBytes>("output_variable_names_as_bytes")
                                  .RemainingArgs()
                                  .RemainingRets());

PYBIND11_MODULE(matlab4jax_cpp, m)
{
    m.def("run_matlab", []()
          { return py::capsule(reinterpret_cast<void *>(run_matlab)); });
}
