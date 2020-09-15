/**
 * @file Network.hpp
 *
 * @brief Backpropagation neural network.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_NETWORK_HPP_
#define _BACKPROPAGATION_NETWORK_HPP_

#include <vector>
#include <iostream>

#include "functions/Activation_function.hpp"
#include "Layer.hpp"
#include "Training_data.hpp"

namespace BackPropagation
{
    /** Class Network */
    class Network
    {
        public:
            /** Class Settings */
            struct Settings
            {
                    uint32_t max_iterations; ///< The maximum number of iterations used to train the network
                    double target_error;     ///< The error at which the network is considered as trained
                    /**
                     * Value between 0 and 1, to trigger a store of the current state
                     * of the network if the error decreases below
                     * (prev_error * store_threshold)
                     */
                    double store_threshold;
                    /** Value equal or greater than 1, to trigger a restore to a
                     * previous state of the network if the error increases beyond
                     * (prev_error * restore_threshold)
                     */
                    double restore_threshold;

                    // Construction
                public:
                    Settings(
                        uint32_t maxIterations,
                        double targetError,
                        double storeThreshold,
                        double restoreThreshold);
            };
        private:
            std::vector<Layer> m_layers;               ///< Network layers
            std::vector<Layer> m_layers_restore_point; ///< Network layers backup

            /**
             * Run a training session on the passed data.
             *
             * @param[in] trainingData Data set to be used in the training process.
             * @param[in] order in which to process the training data.
             * @param[in] settings Network related configuration.
             */
            double iterate(
                const std::vector<Training_data> &trainingData,
                const std::vector<uint32_t> &order,
                const Settings &settings);

            /**
             * Propagate the given inputs through the network.
             *
             * @param[in] inputs to propagate through the network.
             */
            void propagate(const std::vector<double> &inputs);

            /** Store the current network state */
            void save();

            /** Restore a previous network state */
            void restore();

            // Construction
        public:
            /**
             * @param[in] layersInfo Collection containg information about each layer of the network.
             */
            Network(
                std::vector<std::pair<std::uint32_t, functions::Activation_function_cPtr>> layersInfo);
            Network(std::vector<std::uint32_t> layers, functions::Activation_function_cPtr func);
            ~Network();

            // Methods
        public:
            /**
             * Method for triggering a training session.
             *
             * @param[in] trainingData Data set to be used in the training process.
             * @param[in] settings Network related configuration.
             *
             * @return global network error at the end of training.
             */
            double train(const std::vector<Training_data> &trainingData, const Settings &settings);

            /**
             * Method for testing output of the network for a given input.
             *
             * @param[in] input Data set fed to the network.
             *
             * @return output of the network for the given input.
             */
            std::vector<double> test(const std::vector<double> &input);

            friend std::ostream& operator<<(std::ostream &output, const Network &net);
    };
} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_NETWORK_HPP_ */
