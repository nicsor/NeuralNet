/**
 * @file Layer.hpp
 *
 * @brief Basic layer used for backpropagation.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_LAYER_HPP_
#define _BACKPROPAGATION_LAYER_HPP_

#include <stdint.h>

#include <iostream>
#include <vector>

#include "functions/Activation_function.hpp"
#include "Neuron.hpp"

namespace BackPropagation
{
    /** Class layer */
    class Layer
    {
        private:
            std::vector<double> m_output;  ///< Result of the last propagation request
            std::vector<double> m_errors;  ///< Errors to be backpropagated to the input layer
            std::vector<Neuron> m_neurons; ///< Neurons in the current layer

            // Construction
        public:
            /**
             * @param[in] nbrOfNeurons Number of neurons in the current layer.
             * @param[in] nbrOfInputs  Number of incoming connections for the current layer.
             * @param[in] func         Structure containing the activation function and its derivative
             */
            Layer(
                size_t nbrOfNeurons,
                size_t nbrOfInputs,
                functions::Activation_function_cPtr &func);

            // Methods
        public:
            void save();
            void restore();

            /**
             * @return number of neurons in the current layer.
             */
            size_t size() const;

            /**
             * @return the result of the last propagation request.
             */
            const std::vector<double>& output() const;

            /**
             * Set the outputs of the current layer.
             * The intention of the method is to be used for the input layer,
             * for which the neurons shall not perform any computation.
             *
             * @param[in] outputs to be set
             */
            void set_output(const std::vector<double> &outputs);

            /**
             * Propagate the received inputs through the current layer and
             * update the current outputs.
             *
             * @param[in] inputs to propagate
             */
            void propagate(const std::vector<double> &inputs);

            /**
             * Adjust the managed neurons based on the detected error for
             * a given input.
             *
             * @param[in] inputs      used for propapgation
             * @param[in] ouputErrors errors detected for the given inputs
             */
            const std::vector<double>& back_propagate(
                const std::vector<double> &inputs,
                const std::vector<double> &ouputErrors);

            /**
             * Compute the difference between the current output and a given target.
             *
             * @param[in] target to compare the current output against.
             */
            const std::vector<double> compute_errors(const std::vector<double> &target);

            /**
             * Compute the mean error for the whole layer
             *
             * @param[in] expected target output.
             */
            double get_mean_error(const std::vector<double> &expected);

            friend std::ostream& operator<<(std::ostream &output, const Layer &layer);
    };

} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_LAYER_HPP_ */
