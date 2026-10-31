#include "runtime/execution_program.h"
#include <glaze/json/read.hpp>
#include <fstream>
#include <sstream>

namespace GTSim::ExecutionIR {
template <class T> T read(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::invalid_argument("cannot open " + path.string());
    std::string text((std::istreambuf_iterator<char>(file)), {});
    T result;
    constexpr auto options =
        glz::opts{.error_on_unknown_keys = true, .error_on_missing_keys = true};
    if (auto error = glz::read<options>(result, text))
        throw std::invalid_argument(path.string() + ": " +
                                    glz::format_error(error, text));
    return result;
}
Kernel read_kernel(const std::filesystem::path& path) {
    auto k = read<Kernel>(path);
    if (k.record != "ExecKernel" || k.format != "gtsim.exec.kernel/1" ||
        k.schema_version != 1 || k.target != "sm90")
        throw std::invalid_argument(
            "unsupported execution kernel format/target");
    for (const auto& t : k.tensors)
        if (t.record != "KernelTensor")
            throw std::invalid_argument("invalid tensor record");
    for (const auto& c : k.ctas)
        if (c.record != "CTA")
            throw std::invalid_argument("invalid CTA record");
    for (const auto& o : k.ops)
        if (o.record != "ArchInstruction")
            throw std::invalid_argument("invalid instruction record");
    return k;
}
Program read_program(const std::filesystem::path& path) {
    auto p = read<Program>(path);
    if (p.record != "ExecProgram" || p.format != "gtsim.exec.program/1" ||
        p.schema_version != 1)
        throw std::invalid_argument("unsupported execution program format");
    for (const auto& t : p.tensors)
        if (t.record != "KernelTensor")
            throw std::invalid_argument("invalid tensor record");
    for (const auto& l : p.launches)
        if (l.record != "ExecutionLaunch")
            throw std::invalid_argument("invalid launch record");
    for (const auto& c : p.commands)
        if (c.record != "Command")
            throw std::invalid_argument("invalid command record");
    return p;
}
} // namespace GTSim::ExecutionIR
