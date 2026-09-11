#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "lattice.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
    // Onsager's exact result for the square-lattice Ising model,
    // T_c = 2 / ln(1 + sqrt(2)). This is what the scan has to reproduce.
    const double exact_critical_temperature = 2.0 / std::log(1.0 + std::sqrt(2.0));

    struct Measurement
    {
        std::size_t side        = 0;
        double temperature      = 0.0;
        double energy           = 0.0;  // per site
        double magnetisation    = 0.0;  // |m| per site
        double specific_heat    = 0.0;
        double susceptibility   = 0.0;
        double binder_cumulant  = 0.0;
    };

    Measurement run_point(std::size_t side,
                          double temperature,
                          std::size_t equilibration,
                          std::size_t measurements,
                          std::uint64_t seed)
    {
        Lattice lattice(side, seed);
        lattice.randomise();

        const auto sites = static_cast<double>(lattice.sites());

        // One "sweep" is a number of Wolff updates chosen so that a comparable
        // number of spins is touched regardless of lattice size.
        const auto updates_per_sweep = std::max<std::size_t>(1, side / 4);

        for (std::size_t i = 0; i < equilibration; ++i)
            for (std::size_t u = 0; u < updates_per_sweep; ++u)
                lattice.wolff_update(temperature);

        double sum_e = 0.0, sum_e2 = 0.0;
        double sum_m = 0.0, sum_m2 = 0.0, sum_m4 = 0.0;

        for (std::size_t i = 0; i < measurements; ++i)
        {
            for (std::size_t u = 0; u < updates_per_sweep; ++u)
                lattice.wolff_update(temperature);

            const auto e = lattice.energy();
            const auto m = std::abs(lattice.magnetisation());

            sum_e  += e;
            sum_e2 += e * e;
            sum_m  += m;
            sum_m2 += m * m;
            sum_m4 += m * m * m * m;
        }

        const auto n = static_cast<double>(measurements);

        const auto mean_e  = sum_e  / n;
        const auto mean_e2 = sum_e2 / n;
        const auto mean_m  = sum_m  / n;
        const auto mean_m2 = sum_m2 / n;
        const auto mean_m4 = sum_m4 / n;

        Measurement result;
        result.side           = side;
        result.temperature    = temperature;
        result.energy         = mean_e / sites;
        result.magnetisation  = mean_m / sites;

        // C = (<E^2> - <E>^2) / (N T^2),  chi = (<M^2> - <|M|>^2) / (N T)
        result.specific_heat  = (mean_e2 - mean_e * mean_e) / (sites * temperature * temperature);
        result.susceptibility = (mean_m2 - mean_m * mean_m) / (sites * temperature);

        // Binder cumulant U4 = 1 - <m^4> / (3 <m^2>^2). Dimensionless, so
        // curves for different L cross at the critical point.
        result.binder_cumulant = mean_m2 > 0.0
            ? 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
            : 0.0;

        return result;
    }

    // Locate the crossing of two Binder curves by linear interpolation of
    // their difference, which changes sign once near T_c.
    bool crossing(const std::vector<Measurement> & a,
                  const std::vector<Measurement> & b,
                  double & temperature)
    {
        for (std::size_t i = 1; i < a.size() && i < b.size(); ++i)
        {
            const auto previous = a[i - 1].binder_cumulant - b[i - 1].binder_cumulant;
            const auto current  = a[i].binder_cumulant     - b[i].binder_cumulant;

            if (previous == 0.0) { temperature = a[i - 1].temperature; return true; }
            if (previous * current < 0.0)
            {
                const auto t0 = a[i - 1].temperature;
                const auto t1 = a[i].temperature;
                temperature = t0 + (t1 - t0) * previous / (previous - current);
                return true;
            }
        }
        return false;
    }

    void print_usage(const char * program)
    {
        std::cout <<
            "usage: " << program << " [options]\n"
            "\n"
            "  --sizes L,L,...     lattice sides to scan (default 8,16,24,32)\n"
            "  --t-min T           lowest temperature (default 2.0)\n"
            "  --t-max T           highest temperature (default 2.6)\n"
            "  --points N          temperatures in the scan (default 25)\n"
            "  --equilibrate N     equilibration sweeps (default 2000)\n"
            "  --measure N         measurement sweeps (default 8000)\n"
            "  --seed N            base RNG seed\n"
            "  --output DIR        directory for observables.csv (default ./results)\n"
            "  --help              this message\n";
    }
}

int main(int argc, char ** argv)
{
    std::vector<std::size_t> sides{8, 16, 24, 32};

    double t_min = 2.0, t_max = 2.6;
    std::size_t points = 25, equilibration = 2000, measurements = 8000;
    std::uint64_t seed = 92613;
    std::filesystem::path output = "results";

    for (int i = 1; i < argc; ++i)
    {
        const std::string flag = argv[i];
        if (flag == "--help") { print_usage(argv[0]); return EXIT_SUCCESS; }
        if (i + 1 >= argc) { std::cerr << "missing value for " << flag << "\n"; return EXIT_FAILURE; }
        const std::string value = argv[++i];

        try
        {
            if (flag == "--sizes")
            {
                sides.clear();
                std::size_t start = 0;
                while (start <= value.size())
                {
                    const auto comma = value.find(',', start);
                    const auto token = value.substr(start, comma - start);
                    if (!token.empty()) sides.push_back(std::stoul(token));
                    if (comma == std::string::npos) break;
                    start = comma + 1;
                }
            }
            else if (flag == "--t-min")       t_min         = std::stod(value);
            else if (flag == "--t-max")       t_max         = std::stod(value);
            else if (flag == "--points")      points        = std::stoul(value);
            else if (flag == "--equilibrate") equilibration = std::stoul(value);
            else if (flag == "--measure")     measurements  = std::stoul(value);
            else if (flag == "--seed")        seed          = std::stoull(value);
            else if (flag == "--output")      output        = value;
            else { std::cerr << "unknown option " << flag << "\n"; return EXIT_FAILURE; }
        }
        catch (const std::exception &)
        {
            std::cerr << "bad value for " << flag << ": " << value << "\n";
            return EXIT_FAILURE;
        }
    }

    if (sides.empty() || points < 2) { print_usage(argv[0]); return EXIT_FAILURE; }
    std::sort(sides.begin(), sides.end());

    std::vector<double> temperatures(points);
    for (std::size_t i = 0; i < points; ++i)
        temperatures[i] = t_min + (t_max - t_min) * static_cast<double>(i)
                                / static_cast<double>(points - 1);

    // Every (L, T) point is an independent chain, so the scan parallelises
    // perfectly across the flattened grid. Each point gets its own seed, which
    // also makes a run reproducible regardless of thread count.
    std::vector<Measurement> flat(sides.size() * points);

    const auto started = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t job = 0; job < static_cast<std::ptrdiff_t>(flat.size()); ++job)
    {
        const auto s = static_cast<std::size_t>(job) / points;
        const auto t = static_cast<std::size_t>(job) % points;

        flat[static_cast<std::size_t>(job)] =
            run_point(sides[s], temperatures[t], equilibration, measurements,
                      seed + 1000ULL * static_cast<std::uint64_t>(job));
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    try
    {
        std::filesystem::create_directories(output);
        std::ofstream csv(output / "observables.csv");
        if (!csv) throw std::runtime_error("cannot write to " + output.string());

        csv << "L,T,energy_per_site,abs_magnetisation,specific_heat,susceptibility,binder\n";
        csv << std::setprecision(10);

        for (const auto & m : flat)
            csv << m.side << ',' << m.temperature << ',' << m.energy << ','
                << m.magnetisation << ',' << m.specific_heat << ','
                << m.susceptibility << ',' << m.binder_cumulant << '\n';
    }
    catch (const std::exception & exception)
    {
        std::cerr << "error: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "scan: " << sides.size() << " sizes x " << points << " temperatures"
              << "  in " << elapsed << " s\n\n";

    // The estimate: where the Binder curves for successive sizes cross.
    std::vector<std::vector<Measurement>> by_size(sides.size());
    for (std::size_t s = 0; s < sides.size(); ++s)
        by_size[s].assign(flat.begin() + static_cast<std::ptrdiff_t>(s * points),
                          flat.begin() + static_cast<std::ptrdiff_t>((s + 1) * points));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Binder crossings\n";

    std::vector<double> estimates;
    for (std::size_t s = 0; s + 1 < sides.size(); ++s)
    {
        double crossing_temperature = 0.0;
        if (crossing(by_size[s], by_size[s + 1], crossing_temperature))
        {
            estimates.push_back(crossing_temperature);
            std::cout << "  L = " << sides[s] << " x " << sides[s + 1]
                      << "   T_c = " << crossing_temperature << '\n';
        }
    }

    if (!estimates.empty())
    {
        // Crossings drift with L; the largest pair is the best estimate.
        const auto best  = estimates.back();
        const auto error = std::abs(best - exact_critical_temperature)
                         / exact_critical_temperature * 100.0;

        std::cout << "\n  estimate (largest pair) " << best
                  << "\n  exact  2/ln(1+sqrt2)    " << exact_critical_temperature
                  << "\n  relative error          " << std::setprecision(2) << error << " %\n";
    }
    else
    {
        std::cout << "  no crossing in the scanned range\n";
    }

    return EXIT_SUCCESS;
}
