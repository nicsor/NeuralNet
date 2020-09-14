/**
 * @file Training_data.hpp
 *
 * @brief Structure for defining an input entry for a training session.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BACKPROPAGATION_TRAINING_DATA_HPP_
#define _BACKPROPAGATION_TRAINING_DATA_HPP_

#include <vector>

namespace BackPropagation
{
    /** Class Training_data */
    struct Training_data
    {
            std::vector<double> inputs;  ///< Network input
            std::vector<double> outputs; ///< Expected output
    };
} /* namespace BackPropagation */

#endif /* _BACKPROPAGATION_TRAINING_DATA_HPP_ */
