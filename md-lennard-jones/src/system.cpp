#include "system.hpp"

#include <cmath>
#include <ostream>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    // Lennard-Jones pair potential, 4 eps [ (sig/r)^12 - (sig/r)^6 ],
    // evaluated from r^2 to avoid a square root.
    inline double pair_potential(double r2, double sigma, double epsilon) noexcept
    {
        const auto s2  = sigma * sigma / r2;
        const auto s6  = s2 * s2 * s2;
        return 4.0 * epsilon * s6 * (s6 - 1.0);
    }

    // Magnitude of the pair force divided by r, so that
    //   F_vec = pair_force_over_r(r2) * r_vec
    // Derived from -dU/dr: 24 eps / r^2 * [ 2 (sig/r)^12 - (sig/r)^6 ].
    inline double pair_force_over_r(double r2, double sigma, double epsilon) noexcept
    {
        const auto s2  = sigma * sigma / r2;
        const auto s6  = s2 * s2 * s2;
        return 24.0 * epsilon * s6 * (2.0 * s6 - 1.0) / r2;
    }
}

System::System(const Parameters & parameters) : m_parameters(parameters)
{
    const auto cutoff = m_parameters.cutoff * m_parameters.sigma;
    m_cutoff_squared  = cutoff * cutoff;

    // Shifting by U(r_cut) keeps the potential continuous at the cutoff, so
    // total energy is conserved rather than drifting on every crossing.
    m_energy_shift = pair_potential(m_cutoff_squared, m_parameters.sigma, m_parameters.epsilon);

    initialise_fcc();
    compute_forces();

    for (auto & particle : m_particles) particle.m_acceleration = particle.m_new_acceleration;
}

std::size_t System::initialise_fcc()
{
    // An FCC cell holds 4 particles, so n cells per side gives 4 n^3.
    auto cells = static_cast<std::size_t>(
        std::ceil(std::cbrt(static_cast<double>(m_parameters.particles) / 4.0)));
    if (cells == 0) cells = 1;

    const auto cell_size = m_parameters.box_length / static_cast<double>(cells);
    const auto half      = cell_size / 2.0;

    // The four basis sites of the conventional FCC cell.
    const std::array<Particle::vector_t, 4> basis{{
        {0.0,  0.0,  0.0 },
        {half, half, 0.0 },
        {half, 0.0,  half},
        {0.0,  half, half}
    }};

    std::mt19937 engine(m_parameters.seed);

    m_particles.clear();
    m_particles.reserve(4 * cells * cells * cells);

    for (std::size_t i = 0; i < cells; ++i)
    for (std::size_t j = 0; j < cells; ++j)
    for (std::size_t k = 0; k < cells; ++k)
    {
        const Particle::vector_t origin{
            static_cast<double>(i) * cell_size,
            static_cast<double>(j) * cell_size,
            static_cast<double>(k) * cell_size};

        for (const auto & offset : basis)
        {
            Particle::vector_t position{
                origin[0] + offset[0],
                origin[1] + offset[1],
                origin[2] + offset[2]};

            m_particles.emplace_back(
                position,
                Particle::maxwell_boltzmann(engine, m_parameters.temperature, m_parameters.mass));
        }
    }

    // Remove the net centre-of-mass drift, which would otherwise show up as a
    // constant kinetic energy offset and a slowly translating box.
    Particle::vector_t momentum{};
    for (const auto & particle : m_particles)
        for (std::size_t d = 0; d < Particle::dimension; ++d)
            momentum[d] += particle.m_velocity[d];

    const auto count = static_cast<double>(m_particles.size());
    for (auto & particle : m_particles)
        for (std::size_t d = 0; d < Particle::dimension; ++d)
            particle.m_velocity[d] -= momentum[d] / count;

    return m_particles.size();
}

Particle::vector_t System::minimum_image(const Particle::vector_t & a,
                                         const Particle::vector_t & b) const noexcept
{
    const auto length = m_parameters.box_length;

    Particle::vector_t separation{};
    for (std::size_t d = 0; d < Particle::dimension; ++d)
    {
        auto delta = a[d] - b[d];
        delta -= length * std::round(delta / length);
        separation[d] = delta;
    }
    return separation;
}

double System::compute_forces()
{
    const auto count = m_particles.size();

    for (auto & particle : m_particles) particle.m_new_acceleration = {};

    double potential = 0.0;

    // Each thread owns one i and reads every j, so the accumulation into
    // particle i is race-free; the shared potential is handled by reduction.
    #pragma omp parallel for schedule(dynamic, 16) reduction(+ : potential)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(count); ++i)
    {
        auto & particle_i = m_particles[static_cast<std::size_t>(i)];

        for (std::size_t j = 0; j < count; ++j)
        {
            if (static_cast<std::size_t>(i) == j) continue;

            const auto separation = minimum_image(particle_i.m_position, m_particles[j].m_position);

            auto r2 = 0.0;
            for (std::size_t d = 0; d < Particle::dimension; ++d)
                r2 += separation[d] * separation[d];

            if (r2 > m_cutoff_squared || r2 == 0.0) continue;

            const auto force_over_r =
                pair_force_over_r(r2, m_parameters.sigma, m_parameters.epsilon);

            for (std::size_t d = 0; d < Particle::dimension; ++d)
                particle_i.m_new_acceleration[d] += force_over_r * separation[d] / m_parameters.mass;

            // Each unordered pair is visited twice, so halve the contribution.
            potential += 0.5 * (pair_potential(r2, m_parameters.sigma, m_parameters.epsilon)
                                - m_energy_shift);
        }
    }

    return potential;
}

Energies System::update()
{
    const auto dt   = m_parameters.time_step;
    const auto half = 0.5 * dt;

    // Velocity-Verlet, first half: positions from current velocity and
    // acceleration, then velocities advanced half a step.
    for (auto & particle : m_particles)
    {
        for (std::size_t d = 0; d < Particle::dimension; ++d)
        {
            particle.m_position[d] +=
                particle.m_velocity[d] * dt + 0.5 * particle.m_acceleration[d] * dt * dt;

            // Wrap into [0, L) without assuming a single-box displacement.
            particle.m_position[d] = std::fmod(particle.m_position[d], m_parameters.box_length);
            if (particle.m_position[d] < 0.0) particle.m_position[d] += m_parameters.box_length;

            particle.m_velocity[d] += half * particle.m_acceleration[d];
        }
    }

    const auto potential = compute_forces();

    // Second half: complete the velocity update with the new acceleration.
    double kinetic = 0.0;
    for (auto & particle : m_particles)
    {
        auto speed_squared = 0.0;
        for (std::size_t d = 0; d < Particle::dimension; ++d)
        {
            particle.m_velocity[d] += half * particle.m_new_acceleration[d];
            particle.m_acceleration[d] = particle.m_new_acceleration[d];
            speed_squared += particle.m_velocity[d] * particle.m_velocity[d];
        }
        kinetic += 0.5 * m_parameters.mass * speed_squared;
    }

    return Energies{kinetic, potential};
}

void System::write_frame(std::ostream & stream, std::size_t step) const
{
    stream << "ITEM: TIMESTEP\n" << step << '\n';
    stream << "ITEM: NUMBER OF ATOMS\n" << m_particles.size() << '\n';
    stream << "ITEM: BOX BOUNDS pp pp pp\n";
    for (std::size_t d = 0; d < Particle::dimension; ++d)
        stream << 0.0 << ' ' << m_parameters.box_length << '\n';
    stream << "ITEM: ATOMS id x y z\n";

    for (std::size_t i = 0; i < m_particles.size(); ++i)
    {
        const auto & p = m_particles[i];
        stream << i << ' ' << p.m_position[0] << ' '
                            << p.m_position[1] << ' '
                            << p.m_position[2] << '\n';
    }
}
