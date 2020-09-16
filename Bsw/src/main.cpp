/*
 * main.cpp
 *
 * Author: Nicolae Natea
 */

#include <chrono>
#include <iostream>

#include "Network.hpp"

std::vector<Bsw::Training_data> train_data = {
    { { 0, 0, 0, 0 }, { 0, 1, 1, 0 } },
    { { 0, 0, 0, 1 }, { 0, 1, 1, 1 } },
    { { 0, 0, 1, 0 }, { 1, 1, 0, 1 } },
    { { 0, 1, 0, 1 }, { 0, 1, 1, 0 } },
    { { 1, 0, 0, 0 }, { 1, 0, 0, 0 } },
    { { 1, 1, 1, 1 }, { 0, 0, 1, 1 } },
    { { 1, 1, 0, 0 }, { 1, 0, 0, 1 } }
};

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    Bsw::Network net(train_data);
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::milliseconds durationMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

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
            std::cout << (out) << " ";
        }

        std::cout << std::endl;
    }

    return 0;
}
