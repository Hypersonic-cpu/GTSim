#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include <glaze/json/read.hpp>

namespace GTSim::ExecutionIR {

inline constexpr std::uint32_t kStreamProtocolVersion = 1;
inline constexpr std::size_t kDefaultStreamFrameLimit = 1U << 20;

struct Message {
    std::string type;
    std::string json;
};

// Length-prefixed, full-duplex transport used by the persistent frontend
// runner.  The framing layer deliberately knows nothing about Execution IR;
// callers decode one typed payload after each frame.
class FramedTransport {
public:
    explicit FramedTransport(int socket_fd,
                             std::size_t max_frame = kDefaultStreamFrameLimit);
    ~FramedTransport();

    FramedTransport(const FramedTransport&) = delete;
    FramedTransport& operator=(const FramedTransport&) = delete;

    Message receive();
    void send(const Message& message);
    void send_json(std::string_view json);

private:
    int socket_fd_;
    std::size_t max_frame_;
};

template <class T> T decode_message(const Message& message) {
    T result{};
    constexpr auto options =
        glz::opts{.error_on_unknown_keys = false, .error_on_missing_keys = false};
    if (auto error = glz::read<options>(result, message.json))
        throw std::invalid_argument("stream " + message.type + ": " +
                                    glz::format_error(error, message.json));
    return result;
}

std::string json_quote(std::string_view value);
std::string make_typed_json(std::string_view type, std::string_view object);

} // namespace GTSim::ExecutionIR
