/**
 * @file Training_data.hpp
 *
 * @brief Structure for defining an input entry for a training session.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BSW_TRAINING_DATA_HPP_
#define _BSW_TRAINING_DATA_HPP_

#include <vector>

namespace Bsw
{
    /** Class Training_data */
    struct Training_data
    {
            std::vector<int> inputs;  ///< Network input
            std::vector<int> outputs; ///< Expected output
    };
} /* namespace Bsw */

#endif /* _BSW_TRAINING_DATA_HPP_ */
