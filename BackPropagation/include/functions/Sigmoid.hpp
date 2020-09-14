/**
 * @file Sigmoid.hpp
 *
 * @brief Sigmoid activation function.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_FUNCTIONS_SIGMOID_HPP_
#define _BACKPROPAGATION_FUNCTIONS_SIGMOID_HPP_
#include <stdint.h>
#include <math.h>
#include <memory>

#include "functions/Activation_function.hpp"

namespace BackPropagation
{
    namespace functions
    {
        /** Class Sigmoid */
        class Sigmoid : public Activation_function
        {
            public:
                virtual double compute(double x) const
                {
                    return (double) (1.0 / (1.0 + exp(-x)));
                }

                virtual double derivative(double y) const
                {
                    return (double) (y * (1.0 - y));
                }

                virtual ~Sigmoid()
                {
                }
        };

        typedef const std::shared_ptr<const Sigmoid> CSigmoid_cPtr;

    } /* namespace functions */

} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_FUNCTIONS_SIGMOID_HPP_ */
