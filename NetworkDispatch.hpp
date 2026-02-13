// --- LAPLACE NETWORK DISPATCH --- //
// File: NetworkDispatch.hpp
// Description: Routage des paquets réseau + gestion du driver kernel
// Auteur: MasterLaplace

#ifndef NETWORK_DISPATCH_HPP
#define NETWORK_DISPATCH_HPP

#include "lpl_protocol.h"

class WorldPartition; // Forward declaration

/**
 * @brief Ouvre /dev/lpl_driver et mmap le ring buffer partagé.
 *
 * @return Pointeur vers le NetworkRingBuffer mappé, ou nullptr en cas d'erreur.
 */
NetworkRingBuffer *network_init();

/**
 * @brief Libère le mapping mémoire du ring buffer.
 *
 * @param ring Pointeur obtenu via network_init().
 */
void network_cleanup(NetworkRingBuffer *ring);

/**
 * @brief Consomme les paquets du ring buffer et les dispatch dans le WorldPartition.
 *
 * - Entité existante → mise à jour des composants (écriture dans le write buffer).
 * - Entité inconnue → création automatique (insertion dans les deux buffers).
 *
 * @param ring  Ring buffer partagé avec le kernel module (mmap).
 * @param world Le WorldPartition cible.
 */
void network_consume_packets(NetworkRingBuffer *ring, WorldPartition &world);

#endif // NETWORK_DISPATCH_HPP
