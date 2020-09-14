/*
 * Neuron.cpp
 *
 * Author: Nicolae Natea
 */

#include <stdint.h>
#include <assert.h>

#include <random>

#include "Neuron.hpp"

namespace BackPropagation
{
    namespace
    {
        std::random_device g_rand_dev;
        std::mt19937 mt(g_rand_dev());
        std::uniform_real_distribution<double> distrib(-0.5, 0.5);
    }

    Neuron::Neuron(size_t nbrOfInputs, functions::Activation_function_cPtr &func) :
        m_error(0.0f), m_output(0.0f), m_func(func)
    {
        // Check if different than the input layer.
        if (nbrOfInputs)
        {
            m_weights.resize(nbrOfInputs);
            m_momentums.resize(nbrOfInputs);

            for (int i = 0; i < nbrOfInputs; i++)
            {
                // Initialize internal weights with a small random value
                m_weights[i] = distrib(mt);
                m_momentums[i] = 0.0f;
            }
        }
    }

    Neuron::~Neuron()
    {
    }

    double Neuron::compute(const std::vector<double> &inputs)
    {
        size_t offset = 0;
        double sum = 0.0f;

        // Avoid range checks below.
        assert(inputs.size() == m_weights.size());

        for (auto input : inputs)
        {
            sum += m_weights[offset++] * input;
        }

        // Run activation function
        m_output = m_func->compute(sum);
        return m_output;
    }

    void Neuron::adjust(
        double error,
        const std::vector<double> &inputs,
        std::vector<double> &adjustedError)
    {
        // Update error
        m_error = m_func->derivative(m_output) * error;

        // Adjust weights
        {
            size_t offset = 0;

            // Avoid range checks below. m_momentums.size() == m_weights.size()
            assert(inputs.size() == m_weights.size());

            for (auto &output : inputs)
            {
                // Update momentum
                double momentum = m_momentums[offset];
                m_momentums[offset] = output * m_error;

                // Update weights
                m_weights[offset] += m_momentums[offset] + momentum;

                // Compute input errors
                adjustedError[offset] += m_weights[offset] * m_error;

                ++offset;
            }
        }
    }

    std::ostream& operator<<(std::ostream &output, const Neuron &neuron)
    {
        for (auto weight : neuron.m_weights)
        {
            output << weight << " ";
        }

        return output;
    }
}
