/**
 * @file Network.hpp
 *
 * @brief Network based on binary synaptic weights.
 *
 * @author Nicolae Natea
 * Contact: nicu@natea.ro
 */

#ifndef _BSW_NETWORK_HPP_
#define _BSW_NETWORK_HPP_

#include <vector>

#include "Training_data.hpp"

namespace Bsw
{
    struct Node
    {
        std::vector<int> ponderi_Intrare;
        double Threshold;
    };

    /** Class Network */
    class Network
    {
        private:
            int m_inputs_count, m_outputs_count;
            std::vector<std::vector<Node>> nodes;

    	    // Construction
        public:
            Network(const std::vector<Training_data> &data);
            ~Network();

            // Methods
        public:
            std::vector<int> test(const std::vector<int> &input);

        private:
            double train(const std::vector<Training_data> &trainingData);
            void CalculateAverage(const std::vector<Training_data> &trainingData, std::vector<int> &Ave, int& j);
            int compute_average_and_key(const std::vector<Training_data> &trainingData, int j);

            void create_new_plane(
            		const std::vector<Training_data> &trainingData, int j, std::vector<Training_data> &done, int offsetKey);
    };
} /* namespace Bsw */

#endif /* _BSW_NETWORK_HPP_ */
