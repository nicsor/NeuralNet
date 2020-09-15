/*
 * Network.cpp
 *
 * Author: Nicolae Natea
 */

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <numeric>
#include <random>

#include "Network.hpp"

namespace BackPropagation
{
    namespace
    {
        auto rng = std::default_random_engine { };
    }

    Network::Settings::Settings(
        uint32_t maxIterations,
        double targetError,
        double storeThreshold,
        double restoreThreshold) :
            max_iterations(maxIterations),
            target_error(targetError),
            store_threshold(storeThreshold),
            restore_threshold(restoreThreshold)
    {
        // Probably a throw would be more appropriate
        assert(store_threshold >= 0.0 && store_threshold <= 1.0);
        assert(restore_threshold >= 1.0);
    }

    Network::Network(
        std::vector<std::pair<std::uint32_t, functions::Activation_function_cPtr>> layers)
    {
        uint32_t incomingInputs = 0;

        // Populate the current network with layers.
        for (auto layer : layers)
        {
            m_layers.push_back(Layer(layer.first, incomingInputs, layer.second));
            incomingInputs = layer.first;
        }

        // Save the current network state.
        save();
    }

    Network::Network(std::vector<std::uint32_t> layers, functions::Activation_function_cPtr func)
    {
        uint32_t incomingInputs = 0;

        // Populate the current network with layers.
        for (auto nbrOfNeurons : layers)
        {
            m_layers.push_back(Layer(nbrOfNeurons, incomingInputs, func));
            incomingInputs = nbrOfNeurons;
        }

        // Save the current network state.
        save();
    }

    Network::~Network()
    {
    }

    double Network::iterate(
        const std::vector<Training_data> &trainingData,
        const std::vector<uint32_t> &order,
        const Settings &settings)
    {
        auto &outputLayer = m_layers[m_layers.size() - 1];

        for (int index : order)
        {
            const Training_data &data = trainingData[index];

            // Forward propagation.
            propagate(data.inputs);

            // Compute the output error for the current data set.
            std::vector<double> errors = outputLayer.compute_errors(data.outputs);

            // Back-propagate the error starting from the output layer to the input layer.
            // The first layer shall not perform any adjustments.
            for (int index = m_layers.size() - 1; index > 0; index--)
            {
                auto &currLayer = m_layers[index];
                auto &prevLayer = m_layers[index - 1];

                errors = currLayer.back_propagate(prevLayer.output(), errors);
            }
        }

        double averageError = 0.0;

        // Compute average error for all data sets.
        for (const Training_data &data : trainingData)
        {
            propagate(data.inputs);
            averageError += outputLayer.get_mean_error(data.outputs);
        }

        averageError /= trainingData.size();

        return averageError;
    }

    void Network::propagate(const std::vector<double> &inputs)
    {
        auto &inputLayer = m_layers[0];
        auto &outputLayer = m_layers[m_layers.size() - 1];

        // Set the output of the first/input layer.
        inputLayer.set_output(inputs);

        // Forward propagation
        for (uint32_t i = 1; i < m_layers.size(); i++)
        {
            auto &prevLayer = m_layers[i - 1];
            auto &currLayer = m_layers[i];

            currLayer.propagate(prevLayer.output());
        }
    }

    std::vector<double> Network::test(const std::vector<double> &inputs)
    {
        auto &outLayer = m_layers[m_layers.size() - 1];
        propagate(inputs);

        return outLayer.output();
    }

    double Network::train(const std::vector<Training_data> &data, const Settings &settings)
    {
        std::vector<uint32_t> order(data.size());
        std::iota(std::begin(order), std::end(order), 0);

        size_t inputLayerSize = m_layers[0].size();
        size_t outputLayerSize = m_layers[m_layers.size() - 1].size();

        for(auto &trainingData : data) {
            assert(trainingData.inputs.size() == inputLayerSize);
            assert(trainingData.outputs.size() == outputLayerSize);
        }

        // Perform an iteration to get a reference error.
        double error = iterate(data, order, settings);

        // Save network state for which we have the error computed.
        save();

        double previousError = error;
        double storeThreshold = error * settings.store_threshold;
        double restoreThreshold = error * settings.restore_threshold;
        uint32_t interation = 0;

        while (++interation < settings.max_iterations)
        {
            std::shuffle(std::begin(order), std::end(order), rng);
            error = iterate(data, order, settings);

            if (error <= settings.target_error)
            {
                break;
            }
            if (error < storeThreshold)
            {
                // Save the network only when the specified improvement is reached
                previousError = error;
                storeThreshold = error * settings.store_threshold;
                restoreThreshold = error * settings.restore_threshold;
                save();
            }
            else if (error > restoreThreshold)
            {
                // Pretty unlikely with the right data in the current form
                restore();
            }
        }

        if (error < previousError)
        {
            save();
        }
        else if (error > previousError) {
            error = previousError;
            restore();
        }

        return error;
    }

    void Network::save()
    {
        m_layers_restore_point.assign(m_layers.begin(), m_layers.end());
    }

    void Network::restore()
    {
        m_layers.assign(m_layers_restore_point.begin(), m_layers_restore_point.end());
    }

    std::ostream& operator<<(std::ostream &output, const Network &net)
    {
        size_t index = 0;

        output << "Network: ";

        for (auto layer : net.m_layers)
        {
            output << layer.size() << " ";
        }

        for (auto layer : net.m_layers)
        {
            output << "\n\t[Layer " << ++index << "]" << std::endl << layer;
        }

        return output;
    }
}
