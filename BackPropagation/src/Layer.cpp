/*
 * Layer.cpp
 *
 * Author: Nicolae Natea
 */

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <numeric>

#include "Layer.hpp"

namespace BackPropagation
{
    Layer::Layer(size_t nbrOfNeurons, size_t nbrOfInputs, functions::Activation_function_cPtr& activation)
    {
        m_output.resize(nbrOfNeurons);
        m_errors.resize(nbrOfInputs);

        // Populate the current layer with neurons.
        for (size_t i = 0; i < nbrOfNeurons; ++i)
        {
            m_neurons.push_back(Neuron(nbrOfInputs, activation));
        }
    }

    size_t Layer::size() const
    {
        return m_neurons.size();
    }

    void Layer::propagate(const std::vector<double> &inputs)
    {
        size_t index = 0;

        for (auto &neuron : m_neurons)
        {
            // Note: push_back is quite expensive for this repetitive task
            m_output[index++] = neuron.compute(inputs);
        }
    }

    void Layer::set_output(const std::vector<double> &outputs)
    {
        m_output = outputs;
    }

    const std::vector<double>& Layer::output() const
    {
        return m_output;
    }

    const std::vector<double> Layer::compute_errors(const std::vector<double> &targets)
    {
        std::vector<double> errors;
        errors.reserve(size());
        size_t index = 0;

        assert(targets.size() == m_output.size());

        for (auto &target : targets)
        {
            errors.push_back(target - m_output[index++]);
        }

        return errors;
    }

    double Layer::get_mean_error(const std::vector<double> &expected)
    {
        double meanAverageError = 0.0;
        size_t index = 0;

        assert(expected.size() == m_output.size());

        for (double value : expected)
        {
            meanAverageError += abs(value - m_output[index++]);
        }

        return ((double) meanAverageError / (double) size());
    }

    const std::vector<double>& Layer::back_propagate(
        const std::vector<double> &inputs,
        const std::vector<double> &ouputErrors)
    {
        size_t offset = 0;

        // Reset the current value
        std::fill(m_errors.begin(), m_errors.end(), 0);

        // Adjust all the neurons in the current layer.
        for (auto &neuron : m_neurons)
        {
            neuron.adjust(ouputErrors.at(offset++), inputs, m_errors);
        }

        // Return the error for the input layer
        return m_errors;
    }

    std::ostream& operator<<(std::ostream &output, const Layer &layer)
    {
        size_t index = 0;

        for (auto &neuron : layer.m_neurons)
        {
            output << "\t\t[" << ++index << "]: " << neuron << std::endl;
        }

        return output;
    }
}
