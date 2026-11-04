#include "arch_conf.h"
#include "hbf_placement.h"

#include <charconv>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: gtsim_hbf_layout arch.yaml memory.yaml "
                     "tensor=bytes [tensor=bytes ...]\n";
        return 1;
    }
    try {
        auto config = GTSim::make_h100_sxm_variants(argv[1]);
        GTSim::ArchConfYamlReader(argv[2]).parse(config);
        const auto base = config.memory.address_layout.hbf_base;
        const auto& hbf = config.memory.hbf.controller;
        GTSim::HBFMegapageAllocator allocator(base, hbf);
        std::cout << "tensor,logical_begin,logical_end,physical_address,stack,"
                     "die,megaplane,block,page\n";
        for (int i = 3; i < argc; ++i) {
            const std::string_view argument(argv[i]);
            const auto split = argument.find('=');
            if (split == std::string_view::npos || split == 0)
                throw std::invalid_argument("expected tensor=bytes");
            const auto size = argument.substr(split + 1);
            std::uint64_t bytes{};
            const auto end = size.data() + size.size();
            const auto parsed = std::from_chars(size.data(), end, bytes);
            if (parsed.ec != std::errc{} || parsed.ptr != end)
                throw std::invalid_argument("invalid tensor size");
            const auto tensor = allocator.allocate(bytes);
            for (std::uint64_t offset = 0; offset < bytes;) {
                const auto span = tensor.map(offset);
                const auto a = hbfine::decode_address(span.address, base, hbf);
                std::cout << argument.substr(0, split) << ',' << offset << ','
                          << offset + span.bytes << ',' << span.address << ','
                          << a.stack << ',' << a.die << ','
                          << a.plane / hbf.planes_per_command_group << ','
                          << a.block << ',' << a.page << '\n';
                offset += span.bytes;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
