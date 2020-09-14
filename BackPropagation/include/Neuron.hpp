/**
 * @file Neuron.hpp
 *
 * @brief Basic neuron used for backpropagation.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_NEURON_HPP_
#define _BACKPROPAGATION_NEURON_HPP_

#include <iostream>
#include <vector>

#include "functions/Activation_function.hpp"

namespace BackPropagation
{
    /** Class Neuron */
    struct Neuron
    {
        private:
            double m_output;                            ///< Result of the last computation
            double m_error;                             ///< Error according to the last adjustment
            std::vector<double> m_weights;              ///< Input weights
            std::vector<double> m_momentums;            ///< Momentum used for adjusting weights
            functions::Activation_function_cPtr m_func; ///< Neuron activation function

            // Construction
        public:
            /**
             * @param[in] nbrOfInputs Number of inputs
             * @param[in] func        Structure containing the activation function and its derivative
             */
            Neuron(size_t nbrOfInputs, functions::Activation_function_cPtr &func);
            virtual ~Neuron();

            // Methods
        public:
            /**
             * Compute output of neuron
             * @param[in] inputs
             */
            double compute(const std::vector<double> &inputs);

            /**
             * Adjust internal weights for the current neuron
             * @param[in] error          Current error for the current neuron
             * @param[in] inputs         Received parameters
             * @param[out] adjustedError Error to be forwarded to the input layer
             */
            void adjust(
                double error,
                const std::vector<double> &inputs,
                std::vector<double> &adjustedError);

            friend std::ostream& operator<<(std::ostream &output, const Neuron &neuron);
    };
} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_NEURON_HPP_ */
