#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

#include "particle.hpp"

// Energies reported once per step. Total energy is the conserved quantity and
// the correctness check for the integrator.
struct Energies
{
    double kinetic   = 0.0;
    double potential = 0.0;

    double total() const noexcept { return kinetic + potential; }
};

struct Parameters
{
    std::size_t particles   = 500;    // FCC init rounds this to 4 * n^3
    double      box_length  = 10.0;   // reduced units
    double      temperature = 3.0;
    double      time_step   = 0.001;
    double      sigma       = 1.0;
    double      epsilon     = 1.0;
    double      mass        = 1.0;
    double      cutoff      = 2.5;    // in units of sigma
    unsigned    seed        = 20230419;
};

// Lennard-Jones fluid in a cubic box with periodic boundary conditions,
// integrated with velocity-Verlet.
class System
{
public:

    explicit System(const Parameters & parameters);

    // Advance one velocity-Verlet step and return the energies for that step.
    Energies update();

    // Place particles on a face-centred cubic lattice. The particle count is
    // rounded up to the next 4 * n^3; the actual count is returned.
    std::size_t initialise_fcc();

    // Write one frame in the LAMMPS dump format, so trajectories open in OVITO
    // or VMD without conversion.
    void write_frame(std::ostream & stream, std::size_t step) const;

    std::size_t size() const noexcept { return m_particles.size(); }

private:

    // Shortest separation between two positions under the minimum image
    // convention: for each axis, the representative in (-L/2, L/2].
    Particle::vector_t minimum_image(const Particle::vector_t & a,
                                     const Particle::vector_t & b) const noexcept;

    // Accumulate forces and potential energy over all pairs.
    double compute_forces();

    Parameters            m_parameters;
    std::vector<Particle> m_particles;
    double                m_cutoff_squared = 0.0;
    double                m_energy_shift   = 0.0;  // U(r_cut), subtracted so U is continuous
};
