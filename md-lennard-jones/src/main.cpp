#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "system.hpp"

namespace
{
    void print_usage(const char * program)
    {
        std::cout <<
            "usage: " << program << " [options]\n"
            "\n"
            "  --particles N     target particle count, rounded up to 4n^3 (default 500)\n"
            "  --box L           cubic box length in reduced units (default 10)\n"
            "  --temperature T   initial temperature (default 3.0)\n"
            "  --steps N         number of integration steps (default 10000)\n"
            "  --dump-every N    write a trajectory frame every N steps (default 100)\n"
            "  --output DIR      directory for energies.csv and trajectory.lammpstrj\n"
            "                    (default ./results)\n"
            "  --seed N          RNG seed for the initial velocities\n"
            "  --help            this message\n";
    }

    // Minimal argument parsing: enough to make every run reproducible from the
    // command line, without taking a dependency for a handful of flags.
    bool next_value(int argc, char ** argv, int & i, std::string & out)
    {
        if (i + 1 >= argc) return false;
        out = argv[++i];
        return true;
    }
}

int main(int argc, char ** argv)
{
    Parameters parameters;

    std::size_t steps      = 10000;
    std::size_t dump_every = 100;
    std::filesystem::path output = "results";

    for (int i = 1; i < argc; ++i)
    {
        const std::string flag = argv[i];
        std::string value;

        if (flag == "--help") { print_usage(argv[0]); return EXIT_SUCCESS; }

        if (!next_value(argc, argv, i, value))
        {
            std::cerr << "missing value for " << flag << "\n";
            return EXIT_FAILURE;
        }

        try
        {
            if      (flag == "--particles")   parameters.particles   = std::stoul(value);
            else if (flag == "--box")         parameters.box_length  = std::stod(value);
            else if (flag == "--temperature") parameters.temperature = std::stod(value);
            else if (flag == "--steps")       steps                  = std::stoul(value);
            else if (flag == "--dump-every")  dump_every             = std::stoul(value);
            else if (flag == "--seed")        parameters.seed        = std::stoul(value);
            else if (flag == "--output")      output                 = value;
            else
            {
                std::cerr << "unknown option " << flag << "\n";
                print_usage(argv[0]);
                return EXIT_FAILURE;
            }
        }
        catch (const std::exception &)
        {
            std::cerr << "bad value for " << flag << ": " << value << "\n";
            return EXIT_FAILURE;
        }
    }

    try
    {
        std::filesystem::create_directories(output);

        System system(parameters);

        // Both streams are opened once and held open. The original wrote every
        // energy by reopening three files on each of 10,000 steps, which cost
        // more than the force loop it was measuring.
        std::ofstream energies(output / "energies.csv");
        std::ofstream trajectory(output / "trajectory.lammpstrj");

        if (!energies || !trajectory)
            throw std::runtime_error("cannot write to " + output.string());

        energies << "step,kinetic,potential,total\n";
        energies << std::setprecision(10);

        std::cout << "particles: " << system.size()
                  << "  box: "     << parameters.box_length
                  << "  steps: "   << steps << "\n";

        const auto started = std::chrono::steady_clock::now();

        double initial_total = 0.0;
        double max_drift     = 0.0;

        for (std::size_t step = 0; step < steps; ++step)
        {
            const auto energy = system.update();

            if (step == 0) initial_total = energy.total();

            max_drift = std::max(max_drift,
                                 std::abs((energy.total() - initial_total) / initial_total));

            energies << step << ',' << energy.kinetic << ','
                                    << energy.potential << ','
                                    << energy.total() << '\n';

            if (dump_every && step % dump_every == 0)
                system.write_frame(trajectory, step);
        }

        const auto elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();

        std::cout << "elapsed: " << elapsed << " s"
                  << "  (" << elapsed / static_cast<double>(steps) * 1e3 << " ms/step)\n";

        // Relative drift in the total energy is the integrator's correctness
        // check: velocity-Verlet on a shifted potential should stay small.
        std::cout << "max relative energy drift: " << max_drift << "\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception & exception)
    {
        std::cerr << "error: " << exception.what() << "\n";
        return EXIT_FAILURE;
    }
}
