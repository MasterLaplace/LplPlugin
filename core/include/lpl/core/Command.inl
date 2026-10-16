#ifndef LPL_CORE_COMMAND_INL
#define LPL_CORE_COMMAND_INL

namespace lpl::core {

inline void CommandQueue::push(std::unique_ptr<ICommand> command)
{
    lpl::pmr::lock_guard lock{_mutex};
    _commands.push_back(std::move(command));
}

inline void CommandQueue::flush()
{
    lpl::pmr::vector<std::unique_ptr<ICommand>> pending;
    {
        lpl::pmr::lock_guard lock{_mutex};
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

inline bool CommandQueue::empty() const
{
    lpl::pmr::lock_guard lock{_mutex};
    return _commands.empty();
}

} // namespace lpl::core

#endif // LPL_CORE_COMMAND_INL
