/**
 * @file IntrusiveList.hpp
 * @brief Singly-linked intrusive list with zero-allocation insert.
 *
 * The link node is embedded directly inside the user type, meaning
 * no memory is allocated to add an element to the list.  Elements
 * can belong to multiple lists simultaneously if they inherit
 * multiple IntrusiveNode bases.
 *
 * @author MasterLaplace
 * @version 0.1.0
 * @date 2026-02-26
 * @copyright MIT License
 */
#pragma once

#ifndef LPL_MEMORY_INTRUSIVE_LIST_HPP
    #define LPL_MEMORY_INTRUSIVE_LIST_HPP

    #include <lpl/core/Types.hpp>

namespace lpl::memory {

/**
 * @brief Node mixin—inherit this to make a type listable.
 */
struct IntrusiveNode {
    IntrusiveNode *next = nullptr;
};

/**
 * @brief Singly-linked intrusive list.
 * @tparam T Type that publicly inherits IntrusiveNode.
 */
template <typename T>
    requires std::derived_from<T, IntrusiveNode>
class IntrusiveList final {
public:
    /**
     * @brief Push an element to the front of the list.
     * @param element Pointer to the element (must outlive the list entry).
     */
    void pushFront(T *element)
    {
        element->next = _head;
        _head = element;
        ++_size;
    }

    /**
     * @brief Remove an element from the list.
     * @param element Pointer to the element to remove.
     * @return True if the element was found and removed.
     */
    bool remove(T *element)
    {
        IntrusiveNode **pp = &_head;
        while (*pp) {
            if (*pp == element) {
                *pp = element->next;
                element->next = nullptr;
                --_size;
                return true;
            }
            pp = &((*pp)->next);
        }
        return false;
    }

    /**
     * @brief Invoke a callable on each element.
     * @tparam Fn Callable taking T&.
     * @param fn  Visitor.
     */
    template <typename Fn>
    void forEach(Fn &&fn)
    {
        auto *node = _head;
        while (node) {
            fn(*static_cast<T *>(node));
            node = node->next;
        }
    }

    [[nodiscard]] bool        empty() const { return _head == nullptr; }
    [[nodiscard]] core::usize size()  const { return _size; }

    void clear()
    {
        _head = nullptr;
        _size = 0;
    }

private:
    IntrusiveNode *_head = nullptr;
    core::usize    _size = 0;
};

} // namespace lpl::memory

#endif // LPL_MEMORY_INTRUSIVE_LIST_HPP
