#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace GTSim::ExecutionIR {

using TokenId = std::string;

enum class TokenState : unsigned char { Pending, Completed };

// Logical kernel barriers.  Tokens carry no GPU resource and therefore do not
// create a synthetic DAG node or consume a cycle.
class DependencyTokens {
public:
    void declare(const TokenId& id, std::size_t consumers = 0) {
        if (id.empty()) throw std::invalid_argument("empty dependency token");
        auto [it, inserted] = tokens_.emplace(
            id, Entry{TokenState::Pending, consumers});
        if (!inserted && it->second.state == TokenState::Completed)
            throw std::logic_error("dependency token redeclared after completion");
        if (!inserted) it->second.consumers += consumers;
    }

    void add_consumer(const TokenId& id) {
        auto it = tokens_.find(id);
        if (it == tokens_.end()) throw std::logic_error("unknown dependency token");
        ++it->second.consumers;
    }

    bool all_completed(const std::vector<TokenId>& ids) const {
        for (const auto& id : ids) {
            const auto it = tokens_.find(id);
            if (it == tokens_.end() || it->second.state != TokenState::Completed)
                return false;
        }
        return true;
    }

    void complete(const TokenId& id) {
        auto it = tokens_.find(id);
        if (it == tokens_.end()) {
            declare(id);
            it = tokens_.find(id);
        }
        it->second.state = TokenState::Completed;
    }

    void release_if_unreferenced(const TokenId& id) {
        auto it = tokens_.find(id);
        if (it != tokens_.end() && it->second.consumers == 0 &&
            it->second.state == TokenState::Completed)
            tokens_.erase(it);
    }

    void consume(const TokenId& id) {
        auto it = tokens_.find(id);
        if (it == tokens_.end()) throw std::logic_error("unknown dependency token");
        if (it->second.consumers > 0) --it->second.consumers;
        release_if_unreferenced(id);
    }

    std::size_t size() const { return tokens_.size(); }

private:
    struct Entry {
        TokenState state;
        std::size_t consumers;
    };
    std::unordered_map<TokenId, Entry> tokens_;
};

} // namespace GTSim::ExecutionIR
