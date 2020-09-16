/*
 * Network.cpp
 *
 * Author: Nicolae Natea
 */

#include <assert.h>

#include <limits.h>

#include <iostream>
#include <algorithm>
#include <numeric>

#include "Network.hpp"

namespace Bsw {
Network::Network(const std::vector<Training_data> &data) {
	train(data);
}

Network::~Network() {
}

static int get_hamming_distance(const std::vector<int> &x, const std::vector<int> &y) {
	return std::inner_product(x.begin(), x.end(), y.begin(), 0, std::plus<>(), std::not_equal_to<>());
}

double Network::train(const std::vector<Training_data> &data) {
	m_inputs_count = data.at(0).inputs.size();
	m_outputs_count = data.at(0).outputs.size();

	nodes = std::vector<std::vector<Node>>(m_outputs_count);

	std::vector<Training_data> done = data;

	for (int j = 0; j < m_outputs_count; j++) {
		int offsetKey = compute_average_and_key(data, j);
		create_new_plane(data, j, done, offsetKey);

		for (int q = 0; q < done.size(); q++) {
			if (done.at(q).outputs.at(j) == 1) {
				create_new_plane(data, j, done, q);
			}
		}
	}
	return 0;
}

void Network::create_new_plane(const std::vector<Training_data> &trainingData, int j,
		std::vector<Training_data> &done, int offsetKey) {
	std::vector<std::vector<int>> HamDist(1 << m_inputs_count, std::vector<int>(2, 0));
	int maxActiveDist = INT_MIN;
	int minInactiveDist = INT_MAX;
	std::vector<int> key = trainingData.at(offsetKey).inputs;

	for (const Training_data& data : trainingData) {
		int hammingDistance = get_hamming_distance(key, data.inputs);

		if (data.outputs[j] == 1) {
			HamDist[hammingDistance][1]++;
			/* Step 1.3 */
			if (maxActiveDist < hammingDistance) {
				maxActiveDist = hammingDistance;
			}
		} else {
			HamDist[hammingDistance][0]++;
			/* Step 1.4 */
			if (minInactiveDist > hammingDistance) {
				minInactiveDist = hammingDistance;
			}
		}
	}

	/* Step 1.5 */
	int Dist = 0;

	/* Step 2 */
	if (maxActiveDist < minInactiveDist) {
		/* Step 4? */
		done.at(offsetKey).outputs[j] = -1;
	} else {
		Dist = 1;

		/* Step 3? */
		while (!((HamDist[Dist][0] > 0) || (Dist > maxActiveDist))) {
			Dist++;
		}

		for (int l = 0; l < trainingData.size(); l++) {
			int hammingDistance = get_hamming_distance(key, trainingData.at(l).inputs);

			if (hammingDistance < Dist) {
				done.at(l).outputs[j] = -1;
			}
		}

		Dist--;
	}

	// Separation_Plane_Creation
	{
		Node retval;
		int suma = std::accumulate(key.begin(), key.end(), 0);

		retval.Threshold = suma - (double) (Dist + Dist + 1) / 2;

		retval.ponderi_Intrare = std::vector<int>(key.size(), 0);
		for (int i = 0; i < key.size(); i++) {
			// Scale from [0 1] to [-1 1]
			retval.ponderi_Intrare[i] = ((int) key[i] << 1) - 1;
		}

		nodes[j].push_back(retval);
	}

}

int Network::compute_average_and_key(const std::vector<Training_data> &trainingData, int j) {
	/* Step 1.1 */
	std::vector<int> Ave(trainingData.at(0).inputs.size(), 0);
	int keyOffset = INT_MAX;

	// Calculate average
	{
		double impartire;
		int r = 0, p, q, suma;

		for (const Training_data& data : trainingData) {
			r += data.outputs[j];
		}
		impartire = (double) r / 2;
		for (q = 0; q < m_inputs_count; q++) {
			suma = 0;
			for (const Training_data& data : trainingData) {
				suma += (data.inputs[q] * data.outputs[j]);
			}
			Ave[q] = ((double) suma > impartire) ? 1 : 0;
		}
	}

	int min = INT_MAX;

	/* Step 1.2 */
	for (int l = 0; l < trainingData.size(); l++) {
		if (trainingData.at(l).inputs[j] != 0) {
			int hammingDistance = get_hamming_distance(Ave, trainingData.at(l).inputs);

			if (min > hammingDistance) {
				min = hammingDistance;
				keyOffset = l;
			}
		}
	}

	return keyOffset;
}

std::vector<int> Network::test(const std::vector<int> &inputs) {
	std::vector<int> retval(m_outputs_count, 0);

	for (int h = 0; h < m_outputs_count; h++) {
	    for(Node nod: nodes[h]) {
			int sum = std::inner_product(inputs.begin(), inputs.end(),
					nod.ponderi_Intrare.begin(), (int) 0);

			if (sum > nod.Threshold) {
				// activate neuron on output layer
				retval[h] = 1;
				break;
			}
		}
	}

	return retval;
}
}
