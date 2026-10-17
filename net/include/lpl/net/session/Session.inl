#ifndef LPL_NET_SESSION_SESSION_INL
#define LPL_NET_SESSION_SESSION_INL

namespace lpl::net::session {

inline void Session::setAckedSnapshotSeq(core::u64 seq) noexcept
{
    if (seq > _ackedSnapshotSeq)
        _ackedSnapshotSeq = seq;
}

} // namespace lpl::net::session

#endif // LPL_NET_SESSION_SESSION_INL
