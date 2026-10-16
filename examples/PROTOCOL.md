# Toy training gates

Declared before running on 2026-10-14. Correctness/workflow checks only, no benchmark or generalization claim.

Regression: Gaussian data seed37, 64 train/32 heldout examples, deterministic target 3*x0-2*x1+0.5. Linear2->1 initialized seed7, mean MSE, SGD lr0.08, 100 full-batch steps. Pass: heldout MSE below 1e-3.

Classification: two Gaussian clusters centered at -2/+2, standard deviation0.6, data seed111, 128 train/64 heldout examples. Linear2->8/ReLU/Linear8->2 initialized seed11, mean CrossEntropy, Adam lr0.03, 100 full-batch steps. Pass: heldout accuracy at least95% and final training loss below half initial loss.

No parameter tuning against heldout examples. These synthetic tasks validate training and documented examples; they do not establish a PyTorch speed win.
