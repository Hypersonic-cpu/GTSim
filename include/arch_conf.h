#ifndef ARCH_CONF_H
#define ARCH_CONF_H

#include "simulator.h"

#include <glaze/json/generic.hpp>
#include <glaze/yaml.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace glz {

template <> struct meta<GTSim::SchedulePolicy> {
    using enum GTSim::SchedulePolicy;
    static constexpr auto value = enumerate("GTO", GTO, "RR", RR);
};

template <> struct meta<GTSim::WorkloadType> {
    static constexpr auto value = enumerate(
        "GEMM", GTSim::WorkloadType::GEMM, "FA3", GTSim::WorkloadType::FA3,
        "FLASHMLA", GTSim::WorkloadType::FlashMLA, "FLASHDECODING",
        GTSim::WorkloadType::FlashDecoding, "CLUSTERREDUCE",
        GTSim::WorkloadType::ClusterReduce);
};

template <> struct meta<GTSim::NoCTopology> {
    using enum GTSim::NoCTopology;
    static constexpr auto value = enumerate("CROSSBAR", CROSSBAR, "MESH", MESH);
};

template <> struct meta<GTSim::MemoryTopology> {
    using enum GTSim::MemoryTopology;
    static constexpr auto value = enumerate("FullHBM", FullHBM, "FullHBF",
                                              FullHBF, "CoLocated", CoLocated,
                                              "Cascaded", Cascaded);
};

template <> struct meta<GTSim::HBMBackend> {
    using enum GTSim::HBMBackend;
    static constexpr auto value = enumerate("naive", Naive, "ramulator2", Ramulator2);
};

} // namespace glz

namespace GTSim {

// Glaze's YAML map reader keeps the first occurrence of a repeated key. Track
// that condition in the generic tree so the config reader can reject it.
template <class T>
class DuplicateCheckingMap : public std::map<std::string, T, std::less<>> {
    using Base = std::map<std::string, T, std::less<>>;

public:
    using Base::Base;
    using Base::insert;
    using Base::operator[];

    bool has_duplicate_key{};
    std::string duplicate_key;

    template <class Key, class Value> auto emplace(Key&& key, Value&& value) {
        auto result =
            Base::emplace(std::forward<Key>(key), std::forward<Value>(value));
        if (!result.second) {
            has_duplicate_key = true;
            duplicate_key = result.first->first;
        }
        return result;
    }
};

class ArchConfYamlReader {
public:
    explicit ArchConfYamlReader(std::filesystem::path filename)
        : filename_(std::move(filename)) {}

    void parse(SimulatorConfig& conf) const {
        std::ifstream input(filename_);
        if (!input) {
            throw std::runtime_error("cannot open architecture config '" +
                                     filename_.string() + "'");
        }

        const std::string yaml((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
        if (input.bad()) {
            throw std::runtime_error("failed to read architecture config '" +
                                     filename_.string() + "'");
        }

        YamlValue source;
        const auto parse_error = glz::read_yaml(source, yaml);
        if (parse_error) {
            throw std::runtime_error(
                "cannot parse architecture config '" + filename_.string() +
                "': " + glz::format_error(parse_error, yaml));
        }
        if (!source.is_object()) {
            throw std::runtime_error(
                "architecture config '" + filename_.string() +
                "' must contain a YAML mapping at its root");
        }
        if (const auto duplicate = find_duplicate_key(source);
            !duplicate.empty()) {
            throw std::runtime_error("duplicate YAML key '" + duplicate +
                                     "' in '" + filename_.string() + "'");
        }

        YamlValue normalized = normalize(source);
        std::string normalized_yaml;
        const auto write_error = glz::write_yaml(normalized, normalized_yaml);
        if (write_error) {
            throw std::runtime_error(
                "cannot normalize architecture config '" + filename_.string() +
                "': " + glz::format_error(write_error, normalized_yaml));
        }

        // read_yaml updates only members present in the document. Use a copy to
        // keep the caller's config unchanged if a type or key fails validation.
        SimulatorConfig candidate = conf;
        constexpr glz::opts strict_yaml{.error_on_unknown_keys = true};
        const auto config_error =
            glz::read_yaml<strict_yaml>(candidate, normalized_yaml);
        if (config_error) {
            throw std::runtime_error(
                "invalid architecture config '" + filename_.string() +
                "': " + glz::format_error(config_error, normalized_yaml));
        }
        try {
            validate_memory_system_config(candidate.memory);
        } catch (const std::exception& error) {
            throw std::runtime_error("invalid memory configuration in '" +
                                     filename_.string() + "': " + error.what());
        }
        auto& ramulator_file = candidate.memory.hbm.ramulator.config_file;
        if (!ramulator_file.empty() && std::filesystem::path(ramulator_file).is_relative()) {
            ramulator_file = (std::filesystem::absolute(filename_).parent_path() /
                              ramulator_file).lexically_normal().string();
        }
        conf = std::move(candidate);
    }

private:
    using YamlValue =
        glz::generic_json<glz::num_mode::u64, DuplicateCheckingMap>;
    using YamlObject = typename YamlValue::object_t;

    std::filesystem::path filename_;

    static bool is_metadata(std::string_view path) {
        return path == "name" || path == "desc" || path == "description" ||
               path == "source";
    }

    static std::string find_duplicate_key(const YamlValue& value,
                                          const std::string& prefix = {}) {
        if (value.is_object()) {
            const auto& object = value.get<YamlObject>();
            if (object.has_duplicate_key) {
                return prefix.empty() ? object.duplicate_key
                                      : prefix + "." + object.duplicate_key;
            }
            for (const auto& [key, child] : object) {
                const std::string path =
                    prefix.empty() ? key : prefix + "." + key;
                if (auto duplicate = find_duplicate_key(child, path);
                    !duplicate.empty()) {
                    return duplicate;
                }
            }
        } else if (value.is_array()) {
            const auto& array = value.get<typename YamlValue::array_t>();
            for (std::size_t index = 0; index < array.size(); ++index) {
                if (auto duplicate = find_duplicate_key(
                        array[index],
                        prefix + "[" + std::to_string(index) + "]");
                    !duplicate.empty()) {
                    return duplicate;
                }
            }
        }
        return {};
    }

    static bool is_section(std::string_view key) {
        return key == "sram" || key == "l2" || key == "dram" || key == "tma" ||
               key == "tmem" || key == "scheduler" || key == "cluster" ||
               key == "cluster_noc" || key == "noc";
    }

    static bool is_memory_config_field(std::string_view key) {
        return key == "topology" || key == "hbm" || key == "hbf" ||
               key == "d2dlink";
    }

    static std::string member_name(std::string_view section,
                                   std::string_view field) {
        if (section.empty()) {
            if (field == "sm_count") return "num_sms";
            return std::string(field);
        }

        if (section == "sram") return "sram_" + std::string(field);
        if (section == "l2") {
            if (field == "latency_cycles" || field == "hit_latency_cycles") {
                return "l2_hit_latency_cycles";
            }
            return "l2_" + std::string(field);
        }
        if (section == "tma") {
            if (field == "command_issue_interval_cycles")
                return "tma_issue_interval_cycles";
            if (field == "issue_rate_bytes_per_cycle")
                return "tma_issue_rate_bytes_per_cycle";
            return "tma_" + std::string(field);
        }
        if (section == "tmem") return "tmem_" + std::string(field);
        if (section == "scheduler") return std::string(field);
        if (section == "cluster") {
            if (field == "size") return "cluster_size";
            if (field == "noc_bandwidth_bytes_per_cycle") {
                return "cluster_noc_bandwidth_bytes_per_cycle";
            }
            if (field == "noc_latency_cycles")
                return "cluster_noc_latency_cycles";
            return "cluster_" + std::string(field);
        }
        if (section == "cluster_noc")
            return "cluster_noc_" + std::string(field);
        if (section == "noc") return "noc_" + std::string(field);
        return std::string(section) + "." + std::string(field);
    }

    static void normalize_enum_value(std::string_view member,
                                     YamlValue& value) {
        if (member != "schedule_policy" && member != "workload_type" &&
            member != "noc_topology") {
            return;
        }
        auto* text = value.template get_if<std::string>();
        if (text) {
            std::transform(text->begin(), text->end(), text->begin(),
                           [](unsigned char ch) {
                               return static_cast<char>(std::toupper(ch));
                           });
        }
    }

    static void add_member(YamlObject& target, std::string name,
                           const YamlValue& source_value) {
        YamlValue value = source_value;
        normalize_enum_value(name, value);
        const auto [it, inserted] =
            target.emplace(std::move(name), std::move(value));
        if (!inserted) {
            throw std::runtime_error("multiple YAML fields map to '" +
                                     it->first + "'");
        }
    }

    static YamlValue normalize(const YamlValue& source) {
        YamlValue result;
        result.data = YamlObject{};
        auto& target = result.get<YamlObject>();
        YamlValue memory;
        memory.data = YamlObject{};
        auto& memory_target = memory.get<YamlObject>();

        for (const auto& [key, value] : source.get<YamlObject>()) {
            if (is_metadata(key)) continue;

            if (is_memory_config_field(key)) {
                if (key == "topology") {
                    add_member(memory_target, key, value);
                    continue;
                }
                if (!value.is_object() || value.get<YamlObject>().empty()) {
                    throw std::runtime_error("expected a non-empty mapping at '" +
                                             key + "'");
                }
                const auto& section = value.get<YamlObject>();
                if (key == "d2dlink") {
                    for (const auto& [route, route_value] : section) {
                        if (!route_value.is_object() ||
                            route_value.get<YamlObject>().empty()) {
                            throw std::runtime_error(
                                "expected a non-empty mapping at 'd2dlink." +
                                route + "'");
                        }
                        add_member(memory_target, route, route_value);
                    }
                } else {
                    add_member(memory_target, key, value);
                }
                continue;
            }

            // Accept the previous DRAM patch spelling as an HBM alias. A
            // config cannot specify both spellings because they normalize to
            // the same reflected members and are rejected as duplicates.
            if (key == "dram") {
                if (!value.is_object() || value.get<YamlObject>().empty()) {
                    throw std::runtime_error("expected a non-empty mapping at 'dram'");
                }
                YamlValue hbm;
                hbm.data = YamlObject{};
                auto& hbm_target = hbm.get<YamlObject>();
                for (const auto& [field, member_value] : value.get<YamlObject>()) {
                    if (field == "bandwidth_GBps") {
                        add_member(hbm_target, "read_bw_GBps", member_value);
                        add_member(hbm_target, "write_max_bw_GBps", member_value);
                    } else if (field == "latency_cycles" ||
                               field == "miss_penalty_cycles") {
                        add_member(hbm_target, "read_latency_cycles", member_value);
                        add_member(hbm_target, "write_latency_cycles", member_value);
                    } else {
                        throw std::runtime_error("unknown legacy DRAM field '" +
                                                 field + "'");
                    }
                }
                add_member(memory_target, "hbm", hbm);
                continue;
            }

            if (value.is_object() && is_section(key)) {
                const auto& members = value.get<YamlObject>();
                if (members.empty()) {
                    throw std::runtime_error("empty mapping at '" + key + "'");
                }
                for (const auto& [field, member_value] : members) {
                    const std::string path = key + "." + field;
                    if (is_metadata(path)) continue;
                    add_member(target, member_name(key, field), member_value);
                }
            } else {
                add_member(target, member_name({}, key), value);
            }
        }
        if (!memory_target.empty()) {
            add_member(target, "memory", memory);
        }
        return result;
    }
};

static_assert(glz::reflectable<SimulatorConfig>,
              "SimulatorConfig must remain an aggregate for Glaze reflection");

inline SimulatorConfig
make_h100_sxm_variants(const std::filesystem::path& yaml_path) {
    SimulatorConfig conf = make_h100_sxm_config();
    ArchConfYamlReader(yaml_path).parse(conf);
    return conf;
}

} // namespace GTSim

#endif // ARCH_CONF_H
