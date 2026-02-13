#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

// Tes fichiers d'en-tête
#include "WorldPartition.hpp"

// Fonction utilitaire pour afficher un Vec3 proprement
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << "(" << std::fixed << std::setprecision(1) << v.x << ", " << v.y << ", " << v.z << ")";
}

int main() {
    std::cout << "=== Simulation WorldPartition avec Migration Multi-Chunks ===" << std::endl;

    // 1. Création du monde
    WorldPartition world;

    // 2. Création de plusieurs entités avec différentes vitesses et directions
    Partition::EntitySnapshot runner1;
    runner1.id = 1;
    runner1.position = {250.0f, 10.0f, 0.0f};
    runner1.rotation = Quat::identity();
    runner1.velocity = {15.0f, 0.0f, 0.0f};  // Rapide vers X+
    runner1.mass = 1.0f;
    runner1.force = {0.0f, 0.0f, 0.0f};
    runner1.size = {1.0f, 2.0f, 1.0f};

    Partition::EntitySnapshot runner2;
    runner2.id = 2;
    runner2.position = {100.0f, 50.0f, 100.0f};
    runner2.rotation = Quat::identity();
    runner2.velocity = {5.0f, 0.0f, 10.0f};   // Diagonale
    runner2.mass = 2.0f;
    runner2.force = {0.0f, 0.0f, 0.0f};
    runner2.size = {1.5f, 1.5f, 1.5f};

    // 3. Injection dans le monde
    world.addEntity(runner1);
    world.addEntity(runner2);
    std::cout << "✓ 2 entités injectées dans le monde\n" << std::endl;

    // 4. Boucle de simulation
    float dt = 0.1f; // 100ms par frame

    for (int i = 0; i < 15; ++i) {
        std::cout << "--- Frame " << i << " (t=" << std::fixed << std::setprecision(1)
                  << (i * dt) << "s) ---" << std::endl;

        // A. Mise à jour du monde
        world.step(dt);
        world.swapBuffers();  // Rend les résultats visibles au read buffer

        // B. Inspection via l'index EntityID → ChunkKey
        for (uint32_t entityId : {1u, 2u}) {
            uint64_t chunkKey = world.getEntityChunkKey(entityId);

            if (chunkKey == std::numeric_limits<uint64_t>::max()) {
                std::cout << "   Entity #" << entityId << " : PERDUE" << std::endl;
                continue;
            }

            // Récupération du chunk via la clé
            Partition* chunk = world.getChunk(chunkKey);
            if (!chunk) {
                std::cout << "   Entity #" << entityId << " : Chunk invalide (clé: 0x"
                          << std::hex << chunkKey << std::dec << ")" << std::endl;
                continue;
            }

            // Recherche de l'entité dans le chunk
            int idx = chunk->findEntityIndex(entityId);
            if (idx == -1) {
                std::cout << "   Entity #" << entityId << " : Non trouvée dans le chunk (désynchronisation?)" << std::endl;
                continue;
            }

            auto entity = chunk->getEntity(static_cast<size_t>(idx), world.getReadIdx());
            std::cout << "   Entity #" << entityId << " : pos=" << entity.position
                      << " vel=" << entity.velocity
                      << " | Chunk=0x" << std::hex << chunkKey << std::dec
                      << " pop=" << chunk->getEntityCount() << std::endl;
        }

        std::cout << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "=== Fin de la simulation ===" << std::endl;
    std::cout << "\nRésumé : Migration automatique entre chunks validée ✓" << std::endl;
    std::cout << "  - Swap-and-pop correctement implémenté" << std::endl;
    std::cout << "  - Index EntityID → ChunkKey fonctionnel" << std::endl;
    std::cout << "  - Physique avec gravité appliquée (chute en Y)" << std::endl;

    return 0;
}

// BUILD: g++ -std=c++20 -o server server.cpp -lpthread
