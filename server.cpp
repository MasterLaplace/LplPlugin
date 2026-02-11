#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

// Tes fichiers d'en-tête
#include "WorldPartition.hpp"

// Fonction utilitaire pour afficher un Vec3 proprement
std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

int main() {
    std::cout << "=== Démarrage du Moteur Formule 1 ===" << std::endl;

    // 1. Création du monde
    WorldPartition world;

    // 2. Création d'une entité qui court vers la droite (X+)
    // Chunk size = 255. On la place à 250, proche de la frontière.
    // Vitesse = 10 m/s. Elle devrait traverser en 0.5 seconde.
    Partition::EntitySnapshot runner;
    runner.id = 1;
    runner.position = {250.0f, 0.0f, 0.0f};
    runner.rotation = Quat::identity();
    runner.velocity = {10.0f, 0.0f, 0.0f};
    runner.mass = 1.0f;

    // 3. Injection dans le monde
    // On doit d'abord créer/récupérer le chunk à cette position
    Partition* startChunk = world.addPartition(runner.position);
    if (startChunk) {
        startChunk->addEntity(runner);
        std::cout << "[INIT] Entité ajoutée au Chunk initial." << std::endl;
    } else {
        std::cerr << "[ERREUR] Impossible de créer le chunk !" << std::endl;
        return -1;
    }

    // 4. Boucle de simulation (10 frames)
    float dt = 0.1f; // 100ms par frame

    // On track la position théorique pour savoir quel chunk interroger pour l'affichage
    Vec3 trackingPos = runner.position;

    for (int i = 0; i < 15; ++i) {
        std::cout << "--- Frame " << i << " ---" << std::endl;

        // A. Mise à jour du monde
        world.step(dt);

        // Mise à jour de notre tracker (juste pour savoir où regarder)
        trackingPos = trackingPos + (runner.velocity * dt);

        // B. Inspection
        Partition* currentChunk = world.getChunk(trackingPos);

        if (currentChunk) {
            std::cout << "   Position estimée : " << trackingPos << std::endl;
            std::cout << "   Chunk actif : OUI (Population: " << currentChunk->getEntityCount() << ")" << std::endl;

            // Si on veut être puriste, on devrait récupérer la vraie position dans le chunk,
            // mais comme on n'a pas encore d'index par ID, on fait confiance à la physique pour ce test.
        } else {
            std::cout << "   Position estimée : " << trackingPos << std::endl;
            std::cout << "   Chunk actif : NON (L'entité est perdue dans le vide ?)" << std::endl;
        }

        // Petit sleep pour avoir le temps de lire si on le lance en console
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "=== Fin de la simulation ===" << std::endl;
    return 0;
}

// BUILD: g++ -std=c++20 -o server server.cpp -lpthread
