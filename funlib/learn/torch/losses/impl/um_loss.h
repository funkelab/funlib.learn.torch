#include <cstdint>

double c_um_loss_gradient(
	size_t numNodes,
	const double* mst,
	const int64_t* gtSeg,
	double alpha,
	double* gradients,
	double* ratioPos,
	double* ratioNeg,
	double& totalNumPairsPos,
	double& totalNumPairsNeg);

void c_prune_mst(
	size_t numNodes,
	size_t numComponents,
	const double* mst,
	const int64_t* labels,
	const int64_t* components,
	double* filtered_mst);
