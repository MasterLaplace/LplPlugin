/**
 * @file Command.hpp
 * @brief Command pattern implementation for execution decoupling.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-03-05
 * @copyright MIT License
 */

#pragma once

#ifndef LPL_CORE_COMMAND_HPP
#    define LPL_CORE_COMMAND_HPP

#    include <lpl/core/Types.hpp>
#    include <memory>
#    include <mutex>
#    include <vector>

namespace lpl::core {

/**
 * @class ICommand
 * @brief Interface for all executable commands.
 */
class ICommand {
public:
    virtual ~ICommand() = default;

    /** @brief Executes the command encapsulation. */
    virtual void execute() = 0;
};

/**
 * @class CommandQueue
 * @brief Thread-safe queue for buffering and batch-executing commands.
 */
class CommandQueue {
public:
    CommandQueue() = default;
    ~CommandQueue() = default;

    /**
     * @brief Push a new command to the back of the queue.
     * @param command The command to enqueue.
     */
    void push(std::unique_ptr<ICommand> command)
    {
        std::lock_guard lock{_mutex};
        _commands.push_back(std::move(command));
    }

    /**
     * @brief Flushes the queue, executing all pending commands in order.
     */
    void flush()
    {
        std::vector<std::unique_ptr<ICommand>> pending;
        {
            std::lock_guard lock{_mutex};
            pending.swap(_commands);
        }

        for (auto &cmd : pending)
        {
            if (cmd)
            {
                cmd->execute();
            }
        }
    }

    /** @brief Returns true if the queue has no pending commands. */
    [[nodiscard]] bool empty() const
    {
        std::lock_guard lock{_mutex};
        return _commands.empty();
    }

private:
    mutable std::mutex _mutex;
    std::vector<std::unique_ptr<ICommand>> _commands;
};

} // namespace lpl::core

#endif // LPL_CORE_COMMAND_HPP
