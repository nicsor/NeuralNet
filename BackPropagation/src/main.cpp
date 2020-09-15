/*
 * main.cpp
 *
 * Author: Nicolae Natea
 */

#include <chrono>
#include <iostream>

#include "Network.hpp"
#include "functions/Sigmoid.hpp"

std::vector<BackPropagation::Training_data> train_data = {
    { { 0, 0, 0, 0 }, { 1, 1, 1, 1 } },
    { { 0, 0, 0, 1 }, { 1, 1, 1, 0 } },
    { { 0, 0, 1, 0 }, { 1, 1, 0, 1 } },
    { { 0, 0, 1, 1 }, { 1, 1, 0, 0 } },
    { { 0, 1, 0, 0 }, { 1, 0, 1, 1 } },
    { { 0, 1, 0, 1 }, { 1, 0, 1, 0 } },
    { { 0, 1, 1, 0 }, { 1, 0, 0, 1 } },
    { { 0, 1, 1, 1 }, { 1, 0, 0, 0 } },
    { { 1, 0, 0, 0 }, { 0, 1, 1, 1 } },
    { { 1, 0, 0, 1 }, { 0, 1, 1, 0 } },
    { { 1, 0, 1, 0 }, { 0, 1, 0, 1 } },
    { { 1, 0, 1, 1 }, { 0, 1, 0, 0 } },
    { { 1, 1, 0, 0 }, { 0, 0, 1, 1 } },
    { { 1, 1, 0, 1 }, { 0, 0, 1, 0 } },
    { { 1, 1, 1, 0 }, { 0, 0, 0, 1 } },
    { { 1, 1, 1, 1 }, { 0, 0, 0, 0 } }
};

int main()
{
    BackPropagation::functions::Activation_function_cPtr sigmoid =
        std::shared_ptr<const BackPropagation::functions::Activation_function>(
            new BackPropagation::functions::Sigmoid());
    BackPropagation::Network net({ 4, 8, 4 }, sigmoid);
    BackPropagation::Network::Settings settings(10000, 0.01, 0.99, 1);

    auto start = std::chrono::high_resolution_clock::now();
    double error = net.train(train_data, settings);
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds durationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Network training data:" << std::endl << net << std::endl;

    std::cout << "Training error: " << error << std::endl;
    std::cout << "Training duration: " << durationMs.count() << " ms" << std::endl;
    std::cout << "Test trained network:" << std::endl;

    for (auto data : train_data)
    {
        for (auto in : data.inputs)
        {
            std::cout << in << " ";
        }

        std::cout << "= ";

        auto output = net.test(data.inputs);

        for (auto out : output)
        {
            // Round  the result
            std::cout << (int) (out + 0.5f) << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}
