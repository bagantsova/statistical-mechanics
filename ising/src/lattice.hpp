#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

// Square-lattice Ising model with periodic boundaries.
//
// Spins are stored as int8_t rather than bool or int: the update loop is
// memory-bound, and one byte per site keeps a 64x64 lattice inside L1.
class Lattice
{
public:

    Lattice(std::size_t side, std::uint64_t seed) :
        m_side(side), m_sites(side * side), m_spins(side * side, 1), m_engine(seed)
    {
        // Neighbour indices are precomputed once. The modulo arithmetic for
        // periodic wrapping costs more than the table lookup it replaces.
        m_neighbours.resize(4 * m_sites);

        for (std::size_t y = 0; y < m_side; ++y)
        for (std::size_t x = 0; x < m_side; ++x)
        {
            const auto site = index(x, y);
            m_neighbours[4 * site + 0] = index((x + 1) % m_side, y);
            m_neighbours[4 * site + 1] = index((x + m_side - 1) % m_side, y);
            m_neighbours[4 * site + 2] = index(x, (y + 1) % m_side);
            m_neighbours[4 * site + 3] = index(x, (y + m_side - 1) % m_side);
        }
    }

    // Randomise every spin: the infinite-temperature starting configuration.
    void randomise()
    {
        std::bernoulli_distribution coin(0.5);
        for (auto & spin : m_spins) spin = coin(m_engine) ? 1 : -1;
    }

    // One Metropolis sweep: N attempted single-spin flips, sites chosen at
    // random. In 2D the local field takes only five values, so the acceptance
    // probabilities are tabulated instead of calling exp() per site.
    void metropolis_sweep(double temperature)
    {
        std::array<double, 5> accept{};
        for (int k = 0; k < 5; ++k)
        {
            const auto delta_energy = 2.0 * (2 * k - 4);   // -8, -4, 0, 4, 8
            accept[static_cast<std::size_t>(k)] =
                delta_energy <= 0.0 ? 1.0 : std::exp(-delta_energy / temperature);
        }

        std::uniform_int_distribution<std::size_t> site_of(0, m_sites - 1);
        std::uniform_real_distribution<double>     unit(0.0, 1.0);

        for (std::size_t attempt = 0; attempt < m_sites; ++attempt)
        {
            const auto site = site_of(m_engine);

            int neighbour_sum = 0;
            for (std::size_t n = 0; n < 4; ++n)
                neighbour_sum += m_spins[m_neighbours[4 * site + n]];

            // dE = 2 * s_i * sum_j s_j, which lands on one of five values.
            const auto k = (m_spins[site] * neighbour_sum + 4) / 2;

            if (unit(m_engine) < accept[static_cast<std::size_t>(k)])
                m_spins[site] = static_cast<std::int8_t>(-m_spins[site]);
        }
    }

    // One Wolff cluster update. Near the critical point single-spin dynamics
    // suffer critical slowing down (tau ~ L^z with z close to 2); flipping a
    // whole correlated cluster at once keeps autocorrelation times short,
    // which is what makes the Binder crossing measurable at these sizes.
    void wolff_update(double temperature)
    {
        const auto add_probability = 1.0 - std::exp(-2.0 / temperature);

        std::uniform_int_distribution<std::size_t> site_of(0, m_sites - 1);
        std::uniform_real_distribution<double>     unit(0.0, 1.0);

        const auto seed_site = site_of(m_engine);
        const auto original  = m_spins[seed_site];

        m_stack.clear();
        m_stack.push_back(seed_site);
        m_spins[seed_site] = static_cast<std::int8_t>(-original);

        while (!m_stack.empty())
        {
            const auto site = m_stack.back();
            m_stack.pop_back();

            for (std::size_t n = 0; n < 4; ++n)
            {
                const auto neighbour = m_neighbours[4 * site + n];

                if (m_spins[neighbour] == original && unit(m_engine) < add_probability)
                {
                    m_spins[neighbour] = static_cast<std::int8_t>(-original);
                    m_stack.push_back(neighbour);
                }
            }
        }
    }

    // Total energy, E = -J sum_<ij> s_i s_j with J = 1. Each bond is counted
    // once by summing only the two forward neighbours per site.
    double energy() const noexcept
    {
        long long total = 0;

        for (std::size_t site = 0; site < m_sites; ++site)
            total -= m_spins[site] * (m_spins[m_neighbours[4 * site + 0]] +
                                      m_spins[m_neighbours[4 * site + 2]]);

        return static_cast<double>(total);
    }

    double magnetisation() const noexcept
    {
        long long total = 0;
        for (const auto spin : m_spins) total += spin;
        return static_cast<double>(total);
    }

    std::size_t sites() const noexcept { return m_sites; }
    std::size_t side()  const noexcept { return m_side;  }

private:

    std::size_t index(std::size_t x, std::size_t y) const noexcept { return y * m_side + x; }

    std::size_t                m_side;
    std::size_t                m_sites;
    std::vector<std::int8_t>   m_spins;
    std::vector<std::size_t>   m_neighbours;
    std::vector<std::size_t>   m_stack;      // reused across Wolff updates
    std::mt19937_64            m_engine;
};
