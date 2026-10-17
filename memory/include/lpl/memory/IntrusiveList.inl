#ifndef LPL_MEMORY_INTRUSIVE_LIST_INL
#define LPL_MEMORY_INTRUSIVE_LIST_INL

namespace lpl::memory {

template <typename T>
requires std::derived_from<T, IntrusiveNode>
inline void IntrusiveList<T>::pushFront(T *element)
{
    element->next = _head;
    _head = element;
    ++_size;
}

template <typename T>
requires std::derived_from<T, IntrusiveNode>
inline bool IntrusiveList<T>::remove(T *element)
{
    IntrusiveNode **pp = &_head;
    while (*pp)
    {
        if (*pp == element)
        {
            *pp = element->next;
            element->next = nullptr;
            --_size;
            return true;
        }
        pp = &((*pp)->next);
    }
    return false;
}

template <typename T>
requires std::derived_from<T, IntrusiveNode>
template <typename Fn>
inline void IntrusiveList<T>::forEach(Fn &&fn)
{
    auto *node = _head;
    while (node)
    {
        fn(*static_cast<T *>(node));
        node = node->next;
    }
}

template <typename T>
requires std::derived_from<T, IntrusiveNode>
inline void IntrusiveList<T>::clear()
{
    _head = nullptr;
    _size = 0;
}

} // namespace lpl::memory

#endif // LPL_MEMORY_INTRUSIVE_LIST_INL
