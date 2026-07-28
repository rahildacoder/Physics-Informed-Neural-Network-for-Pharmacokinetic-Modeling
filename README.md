# Predicting Drug Absorption Using Physics-Informed Neural Networks

A course project (APMA 2070, Brown University) modeling pharmacokinetic drug absorption dynamics with Physics-Informed Neural Networks (PINNs), following the methodology of Yazdani et al. (2020).

## Overview

Drug pharmacokinetics — the rate at which a drug is absorbed, distributed, and eliminated in the body — is commonly modeled using compartment models. This project implements a **three-compartment model** representing the gastrointestinal (GI) tract, bloodstream, and urinary tract, and solves it both numerically and with PINNs, in both the forward and inverse directions.

The three-compartment system is given by:

```
dG/dt = -k_g * G,              G(0) = G0
dB/dt =  k_g * G - k_b * B,    B(0) = 0
dU/dt =  k_b * B,              U(0) = 0
```

where `G(t)`, `B(t)`, and `U(t)` denote the drug amount in the GI tract, bloodstream, and urinary tract respectively, and `k_g`, `k_b` are absorption/elimination rate constants. Parameters follow Barnes & Fulford (2011) for the antibiotic tetracycline:

- `k_g = 0.72 hour⁻¹`
- `k_b = 0.15 hour⁻¹`
- `G0 = 1e-4 mg`

## Tasks

1. **Forward solution** — Solved the ODE system using `scipy.integrate.solve_ivp` (RK45) as ground truth, sampled over `t = 0–48` hours (2000 points).

2. **Forward PINN** — Implemented a PINN following Yazdani et al.'s methodology: an input-scaling layer (`t → t/T`), a feature layer encoding two exponential-decay features plus normalized time, two hidden layers of 32 neurons with tanh activation, and an output-scaling layer. Trained with a combined data + ODE-residual + auxiliary loss. Achieved a normalized MSE of **1.494e-6**.

3. **Sample-size sensitivity (forward)** — Swept training sample size from 10 to 400. Found no consistent MSE decrease beyond a certain threshold, suggesting further gains require better optimization, architecture, or loss weighting rather than more data.

4. **Inverse PINN** — Treated `k_g` and `k_b` as trainable parameters (initialized at 0.5) and recovered them from the full trajectory. Recovered `k_g = 0.717562` (true: 0.72, error 2.44e-3) and `k_b = 0.150230` (true: 0.15, error 2.30e-4) after 20,000 epochs.

5. **Identifiability with partial observations** — Repeated the inverse problem using only `B(t)` observations. Parameters remained identifiable, though with somewhat higher error: `k_g` error 5.75e-3, `k_b` error 1.43e-4 after 50,000 epochs.

6. **Uncertainty quantification** — Used deep ensembles (10 members per noise level) under Gaussian noise with standard deviations of 0.01, 0.02, 0.05, 0.1, and 0.2, to study how parameter and trajectory uncertainty scale with observation noise.

7. **Adaptive activation functions** — Tested adaptive-slope tanh activations (`tanh(σx)`) with initial slopes 1.0, 2.0, and 3.0 for the inverse problem. Best performance at `σ = 2.0`, reducing normalized MSE from 3.659e-6 to 8.658e-7. Improvement was sensitive to initialization.

8. **Adaptive loss weighting** — Dynamically reweighted the data and auxiliary loss terms relative to the ODE-residual loss using gradient-magnitude ratios, smoothed over training. Reduced `k_g` error by 1.86×, `k_b` error by 4.64×, and normalized MSE by 7.73× versus fixed weights — but introduced less stable training curves, indicating the technique mitigates optimization-related ill-conditioning without fully resolving the inverse problem's underlying ill-posedness.

## Model Architecture

- **Input**: normalized time `t̃ = t / T`
- **Features**: `t̃`, `exp(-k_g · T · t̃)`, `exp(-k_b · T · t̃)`
- **Network**: 2 hidden layers, 32 neurons each, tanh activation
- **Output**: 3 values (G, B, U), rescaled back to physical units
- **Loss**: `L = λ_data · L_data + λ_ODE · L_ODE + λ_aux · L_aux`
  - `L_data`: MSE between predicted and ground-truth trajectories
  - `L_ODE`: ODE-residual loss at collocation points
  - `L_aux`: initial-condition and `t = 24h` boundary enforcement

## Results Summary

| Task | Metric | Value |
|---|---|---|
| Forward PINN | Normalized MSE | 1.494e-6 |
| Inverse PINN (full obs.) | `k_g` error | 2.438e-3 |
| Inverse PINN (full obs.) | `k_b` error | 2.296e-4 |
| Inverse PINN (`B(t)`-only obs.) | `k_g` error | 5.745e-3 |
| Inverse PINN (`B(t)`-only obs.) | `k_b` error | 1.426e-4 |
| Adaptive activation (best, σ=2.0) | Normalized MSE | 8.658e-7 |
| Adaptive loss weighting | Normalized MSE | 4.732e-7 (7.73× lower than fixed) |

## Tech Stack

Python, PyTorch, SciPy, NumPy, Matplotlib

## References

1. B. Barnes and G. R. Fulford, *Mathematical Modelling with Case Studies*, Chapman and Hall/CRC, 2011.
2. O. Egbelowo, "Nonlinear elimination of drugs in one-compartment pharmacokinetic models," *Mathematical and Computational Applications*, 23(2):27, 2018.
3. K. Goswami et al., "Study of drug assimilation using physics-informed neural networks," *International Journal of Information Technology*, 2022.
4. A. Heck et al., "Modelling intake and clearance of alcohol in humans," *Electronic Journal of Mathematics and Technology*, 1(3):232–244, 2007.
5. M. Khanday et al., "Mathematical models for drug diffusion through blood and tissue compartments," *Alexandria Journal of Medicine*, 53(3):245–249, 2017.
6. G. Koch-Noble, "Drugs in the classroom: Using pharmacokinetics to introduce biomathematical modeling," *Mathematical Modelling of Natural Phenomena*, 6(6):227–244, 2011.
7. E. Spitznagel, "Two-compartment pharmacokinetic models," *C-ODE-E Newsletter*, Harvey Mudd College, 1992.
8. A. Yazdani et al., "Systems biology informed deep learning for inferring parameters and hidden dynamics," *PLoS Computational Biology*, 16(11):e1007575, 2020.

## Course

Project completed for **APMA 2070**, Brown University (course project contributed by Dr. He Li, School of Engineering).
