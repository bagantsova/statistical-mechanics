#pragma once

#include <array>
#include <cmath>
#include <random>

// A single particle in the Lennard-Jones fluid.
//
// Positions, velocities and accelerations are fixed-size arrays rather than
// std::vector: the dimension is a compile-time constant, and keeping the data
// inline removes one indirection per component from the inner force loop.
class Particle
{
public:

    static constexpr std::size_t dimension = 3;

    using vector_t = std::array<double, dimension>;

    Particle() noexcept = default;

    explicit Particle(const vector_t & position, const vector_t & velocity = {}) noexcept :
        m_position(position), m_velocity(velocity) {}

    // Draw a velocity from the Maxwell-Boltzmann distribution at temperature T.
    // Each Cartesian component is normal with variance kT/m.
    static vector_t maxwell_boltzmann(std::mt19937 & engine,
                                      double temperature,
                                      double mass = 1.0,
                                      double boltzmann_constant = 1.0)
    {
        std::normal_distribution<double> normal(
            0.0, std::sqrt(boltzmann_constant * temperature / mass));

        vector_t velocity{};
        for (auto & component : velocity) component = normal(engine);
        return velocity;
    }

    vector_t m_position{};
    vector_t m_velocity{};
    vector_t m_acceleration{};
    vector_t m_new_acceleration{};
};
