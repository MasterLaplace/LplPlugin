/**
 * @file EntityRegistry.hpp
 * @brief Registre central d'entités — sparse set + chunk routing.
 *
 * Pont entre les IDs publics (réseau) et la localisation physique (chunk Morton key).
 * Remplace à la fois _entityToChunk (WorldPartition) et la partie gestion d'entités de Core (Engine).
 *
 * Fournit :
 * - Lookup O(1) publicId → chunkKey
 * - IDs générationnels pour invalidation sûre des handles
 * - Free list pour recyclage des slots
 *
 * @author MasterLaplace
 */

#pragma once

#include <cstdint>
#include <atomic>
#include <vector>

#ifndef MAX_ENTITIES
#define MAX_ENTITIES 10000
#endif
#ifndef MAX_ID
#define MAX_ID 1000000
#endif
#ifndef INDEX_BITS
#define INDEX_BITS 14
#endif
#ifndef INDEX_MASK
#define INDEX_MASK 0x3FFF
#endif

class EntityRegistry {
public:
    static constexpr uint32_t INVALID_SLOT  = UINT32_MAX;
    static constexpr uint64_t INVALID_CHUNK = UINT64_MAX;

    EntityRegistry()
    {
        _sparseToSlot.assign(MAX_ID, INVALID_SLOT);

        uint32_t val = MAX_ENTITIES;
        for (uint32_t i = 0u; i < MAX_ENTITIES; ++i)
        {
            _freeSlots[i] = --val;
            _generations[i] = 0u;
            _chunkKeys[i] = INVALID_CHUNK;
            _slotToPublic[i] = INVALID_SLOT;
        }
        _freeCount.store(MAX_ENTITIES, std::memory_order_relaxed);
    }

    // ─── Enregistrement / Désenregistrement ───────────────────────

    /**
     * @brief Enregistre une nouvelle entité dans le registre.
     * @param publicId Identifiant public (réseau, ex: 42).
     * @param chunkKey Clé Morton du chunk contenant l'entité.
     * @return Smart handle (generation << INDEX_BITS | slot), ou UINT32_MAX en cas d'échec.
     */
    uint32_t registerEntity(const uint32_t publicId, const uint64_t chunkKey) noexcept
    {
        if (publicId >= MAX_ID)
            return UINT32_MAX;

        uint32_t fc = _freeCount.fetch_sub(1u, std::memory_order_relaxed);
        if (fc == 0u || fc - 1u >= MAX_ENTITIES)
        {
            _freeCount.fetch_add(1u, std::memory_order_relaxed);
            return UINT32_MAX;
        }

        uint32_t slot = _freeSlots[fc - 1];
        _chunkKeys[slot] = chunkKey;
        _slotToPublic[slot] = publicId;
        _sparseToSlot[publicId] = slot;

        return (static_cast<uint32_t>(_generations[slot]) << INDEX_BITS) | slot;
    }

    /**
     * @brief Désenregistre une entité par son ID public.
     * Incrémente la génération pour invalider les handles existants.
     */
    void unregisterEntity(const uint32_t publicId) noexcept
    {
        if (publicId >= MAX_ID) return;

        uint32_t slot = _sparseToSlot[publicId];
        if (slot >= MAX_ENTITIES) return;

        _sparseToSlot[publicId] = INVALID_SLOT;
        _chunkKeys[slot] = INVALID_CHUNK;
        _slotToPublic[slot] = INVALID_SLOT;
        _generations[slot]++;

        uint32_t pos = _freeCount.fetch_add(1u, std::memory_order_relaxed);
        if (pos < MAX_ENTITIES)
            _freeSlots[pos] = slot;
    }

    // ─── Localisation ─────────────────────────────────────────────

    /**
     * @brief Localise une entité par son ID public.
     * @return Clé Morton du chunk contenant l'entité, ou INVALID_CHUNK si non trouvée.
     */
    [[nodiscard]] uint64_t getChunkKey(const uint32_t publicId) const noexcept
    {
        if (publicId >= MAX_ID) return INVALID_CHUNK;
        uint32_t slot = _sparseToSlot[publicId];
        if (slot >= MAX_ENTITIES) return INVALID_CHUNK;
        return _chunkKeys[slot];
    }

    /**
     * @brief Met à jour la clé de chunk d'une entité (après migration inter-chunk).
     */
    void updateChunkKey(const uint32_t publicId, const uint64_t newChunkKey) noexcept
    {
        if (publicId >= MAX_ID) return;
        uint32_t slot = _sparseToSlot[publicId];
        if (slot >= MAX_ENTITIES) return;
        _chunkKeys[slot] = newChunkKey;
    }

    // ─── Validation ───────────────────────────────────────────────

    /**
     * @brief Vérifie la validité d'un smart handle (génération correcte).
     */
    [[nodiscard]] bool isValid(const uint32_t smartId) const noexcept
    {
        uint32_t slot = smartId & INDEX_MASK;
        uint16_t gen = static_cast<uint16_t>(smartId >> INDEX_BITS);
        if (slot >= MAX_ENTITIES) return false;
        return _generations[slot] == gen;
    }

    /**
     * @brief Vérifie si un ID public est actuellement enregistré.
     */
    [[nodiscard]] bool isRegistered(const uint32_t publicId) const noexcept
    {
        if (publicId >= MAX_ID) return false;
        return _sparseToSlot[publicId] != INVALID_SLOT;
    }

    // ─── Utilitaires ──────────────────────────────────────────────

    [[nodiscard]] static uint32_t getSlot(const uint32_t smartId) noexcept
    {
        return smartId & INDEX_MASK;
    }

    [[nodiscard]] static uint16_t getGeneration(const uint32_t smartId) noexcept
    {
        return static_cast<uint16_t>(smartId >> INDEX_BITS);
    }

private:
    std::vector<uint32_t> _sparseToSlot;        ///< publicId → slot (heap, ~4MB)
    uint64_t  _chunkKeys[MAX_ENTITIES];          ///< slot → chunkKey Morton
    uint16_t  _generations[MAX_ENTITIES];        ///< slot → génération (pour invalidation)
    uint32_t  _slotToPublic[MAX_ENTITIES];       ///< slot → publicId (reverse mapping)
    uint32_t  _freeSlots[MAX_ENTITIES];          ///< pile des slots libres
    std::atomic<uint32_t> _freeCount;            ///< compteur atomique de slots libres
};
