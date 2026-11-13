#include "arch_conf.h"
#include "runtime/streaming_program_runner.h"
#include <glaze/json/write.hpp>
#include <charconv>
#include <cstddef>
#include <limits>
#include <string_view>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 5 || std::string_view(argv[2]) != "--memory-config") {
        std::cerr
            << "Usage: gtsim_program architecture.yaml --memory-config "
               "memory.yaml --stream-fd FD [--max-cycles N] [--max-frame N]\n";
        return 1;
    }
    try {
        int max_cycles = 100000000;
        std::size_t max_frame = GTSim::ExecutionIR::kDefaultStreamFrameLimit;
        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        GTSim::ArchConfYamlReader(argv[3]).parse(config);
        config.silence_mode = true;
        if (std::string_view(argv[4]) == "--stream-fd") {
            if (argc < 6)
                throw std::invalid_argument("--stream-fd needs an fd");
            int fd = -1;
            const std::string_view value(argv[5]);
            const auto end = value.data() + value.size();
            const auto [ptr, error] = std::from_chars(value.data(), end, fd);
            if (ptr != end || error != std::errc{} || fd < 0)
                throw std::invalid_argument(
                    "--stream-fd must be a nonnegative int");
            for (int i = 6; i < argc; ++i) {
                if (i + 1 >= argc)
                    throw std::invalid_argument("invalid stream runner option");
                const std::string_view option(argv[i]);
                const std::string_view value(argv[++i]);
                if (option == "--max-cycles") {
                    const auto cycles_end = value.data() + value.size();
                    const auto parsed =
                        std::from_chars(value.data(), cycles_end, max_cycles);
                    if (parsed.ptr != cycles_end || parsed.ec != std::errc{} ||
                        max_cycles <= 0)
                        throw std::invalid_argument(
                            "--max-cycles must be a positive int");
                } else if (option == "--max-frame") {
                    unsigned long long parsed_frame = 0;
                    const auto frame_end = value.data() + value.size();
                    const auto parsed =
                        std::from_chars(value.data(), frame_end, parsed_frame);
                    if (parsed.ptr != frame_end || parsed.ec != std::errc{} ||
                        parsed_frame == 0 ||
                        parsed_frame >
                            std::numeric_limits<std::uint32_t>::max())
                        throw std::invalid_argument(
                            "--max-frame must be a positive uint32");
                    max_frame = static_cast<std::size_t>(parsed_frame);
                } else {
                    throw std::invalid_argument("invalid stream runner option");
                }
            }
            GTSim::ExecutionIR::FramedTransport transport(fd, max_frame);
            auto timing = GTSim::ExecutionIR::StreamingProgramRunner(
                              config, transport, max_cycles)
                              .run();
            std::cout << "Program timing JSON: "
                      << glz::write_json(timing).value() << '\n';
            return 0;
        }
        throw std::invalid_argument("streaming execution requires --stream-fd");
    } catch (const std::exception& e) {
        std::cerr << "Execution program error: " << e.what() << '\n';
        return 2;
    }
}
