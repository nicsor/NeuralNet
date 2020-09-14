/**
 * @file Activation_function.hpp
 *
 * @brief Common interface for an activation function.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_FUNCTIONS_ACTIVATION_FUNCTION_HPP
#define _BACKPROPAGATION_FUNCTIONS_ACTIVATION_FUNCTION_HPP

#include <memory>

namespace BackPropagation
{
    namespace functions
    {
        /** Activation_function interface */
        class Activation_function
        {
            public:
                /**
                 * Activation function
                 *
                 * @param[in] x sum of all inputs adjusted
                 *    according to the neuron's internal weights.
                 */
                virtual double compute(double x) const = 0;

                /**
                 * Activation function derivative
                 *
                 * @param[in] y result of the the activation function.
                 */
                virtual double derivative(double y) const = 0;

                virtual ~Activation_function()
                {
                }
        };

        typedef std::shared_ptr<const Activation_function> Activation_function_cPtr;

    } /* namespace functions */

} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_FUNCTIONS_ACTIVATION_FUNCTION_HPP */
