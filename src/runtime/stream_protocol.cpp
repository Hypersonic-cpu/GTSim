#include "runtime/stream_protocol.h"

#include <cerrno>
#include <cstring>
#include <sys/socket.h>
#include <unistd.h>

#include <glaze/json/read.hpp>

namespace GTSim::ExecutionIR {
namespace {
struct Envelope {
    std::string type;
};

void read_all(int fd, void* destination, std::size_t bytes) {
    auto* out = static_cast<char*>(destination);
    std::size_t offset = 0;
    while (offset < bytes) {
        const auto result = ::recv(fd, out + offset, bytes - offset, 0);
        if (result == 0) throw std::runtime_error("stream transport EOF");
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("stream receive failed: ") +
                                     std::strerror(errno));
        }
        offset += static_cast<std::size_t>(result);
    }
}

void write_all(int fd, const void* source, std::size_t bytes) {
    const auto* input = static_cast<const char*>(source);
    std::size_t offset = 0;
    while (offset < bytes) {
        const auto result = ::send(fd, input + offset, bytes - offset,
                                   MSG_NOSIGNAL);
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(std::string("stream send failed: ") +
                                     std::strerror(errno));
        }
        offset += static_cast<std::size_t>(result);
    }
}

std::uint32_t read_length(const std::array<std::uint8_t, 4>& bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) |
           (static_cast<std::uint32_t>(bytes[3]) << 24);
}

void write_length(std::array<std::uint8_t, 4>& bytes, std::uint32_t value) {
    bytes[0] = static_cast<std::uint8_t>(value);
    bytes[1] = static_cast<std::uint8_t>(value >> 8);
    bytes[2] = static_cast<std::uint8_t>(value >> 16);
    bytes[3] = static_cast<std::uint8_t>(value >> 24);
}
} // namespace

FramedTransport::FramedTransport(int socket_fd, std::size_t max_frame)
    : socket_fd_(socket_fd), max_frame_(max_frame) {
    if (socket_fd_ < 0 || max_frame_ == 0 ||
        max_frame_ > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("invalid stream transport configuration");
}

FramedTransport::~FramedTransport() {
    if (socket_fd_ >= 0) ::close(socket_fd_);
}

Message FramedTransport::receive() {
    std::array<std::uint8_t, 4> header{};
    read_all(socket_fd_, header.data(), header.size());
    const auto length = read_length(header);
    if (length == 0 || length > max_frame_)
        throw std::invalid_argument("stream frame exceeds configured limit");
    std::string json(length, '\0');
    read_all(socket_fd_, json.data(), json.size());
    Envelope envelope{};
    constexpr auto options =
        glz::opts{.error_on_unknown_keys = false, .error_on_missing_keys = false};
    if (auto error = glz::read<options>(envelope, json))
        throw std::invalid_argument("invalid stream JSON: " +
                                    glz::format_error(error, json));
    if (envelope.type.empty())
        throw std::invalid_argument("stream message has no type");
    return {std::move(envelope.type), std::move(json)};
}

void FramedTransport::send(const Message& message) { send_json(message.json); }

void FramedTransport::send_json(std::string_view json) {
    if (json.empty() || json.size() > max_frame_ || json.size() >
                                               std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("stream frame exceeds configured limit");
    std::array<std::uint8_t, 4> header{};
    write_length(header, static_cast<std::uint32_t>(json.size()));
    write_all(socket_fd_, header.data(), header.size());
    write_all(socket_fd_, json.data(), json.size());
}

std::string json_quote(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const auto ch : value) {
        switch (ch) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                constexpr char hex[] = "0123456789abcdef";
                result += "\\u00";
                result.push_back(hex[(static_cast<unsigned char>(ch) >> 4) & 0xf]);
                result.push_back(hex[static_cast<unsigned char>(ch) & 0xf]);
            } else {
                result.push_back(ch);
            }
        }
    }
    result.push_back('"');
    return result;
}

std::string make_typed_json(std::string_view type, std::string_view object) {
    if (object.empty() || object.front() != '{' || object.back() != '}')
        throw std::invalid_argument("stream payload must be a JSON object");
    std::string result = "{\"type\":" + json_quote(type);
    if (object.size() > 2) {
        result.push_back(',');
        result.append(object.substr(1, object.size() - 2));
    }
    result.push_back('}');
    return result;
}
} // namespace GTSim::ExecutionIR
