# emulator-zoo

*"Please do not feed the models" - But feel free to train them!*

![Image](https://res.cloudinary.com/df9pocq2o/image/upload/v1763490991/emulator-zoo_akhtez.png)

*A comprehensive collection of cosmological emulators for high-precision theoretical predictions*

[![License](https://img.shields.io/github/license/CosmologicalEmulators/emulator-zoo)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple)](https://julialang.org)
[![Status](https://img.shields.io/badge/Status-Active-green)](https://github.com/CosmologicalEmulators/emulator-zoo)

## 🎯 Overview

Welcome to **emulator-zoo**, the centralized repository for state-of-the-art cosmological emulators within the [CosmologicalEmulators](https://github.com/CosmologicalEmulators) organization. This repository provides a unified framework for training, validating, and deploying neural network emulators for various cosmological observables.

### Key Features

- 🚀 **Production-Ready Emulators**: CMB power spectra (Cℓ) and galaxy power spectrum multipoles
- 🔬 **Multiple Cosmologies**: ΛCDM, w₀wₐCDM, massive neutrinos, and axion models
- 📊 **Validated Performance**: Comprehensive validation across extensive cosmological parameter spaces
- 🔧 **Ensemble Methods**: Support for Phalanx.jl ensemble averaging with 50-75% outlier reduction
- 📦 **Two Backend Frameworks**: Capse.jl (CMB) and Effort.jl (EFTofLSS)
- ♻️ **Full Reproducibility**: Complete training pipelines with data generation scripts

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Available Emulators](#-available-emulators)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Validation & Performance](#-validation--performance)
- [Advanced Features](#-advanced-features)
- [Contributing](#-contributing)
- [Citation](#-citation)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/CosmologicalEmulators/emulator-zoo.git
cd emulator-zoo

# Set up Julia environment
julia --project=.
julia> using Pkg; Pkg.instantiate()

# Download validation dataset (required for testing)
# TODO: Add download instructions once validation data is hosted
# wget https://path-to-validation-data/class_mnuw0wacdm_validation.tar.gz
# tar -xzf class_mnuw0wacdm_validation.tar.gz

# Run a quick test (example with Capse.jl)
# Note: Requires validation dataset to be downloaded first
julia> include("Capse.jl/class_mnuw0wacdm/validator.jl")
```

## 🗂️ Available Emulators

### Capse.jl Framework (CMB Power Spectra)

| Emulator | Cosmology | Observables | Parameters | Status |
|----------|-----------|-------------|------------|--------|
| `class_lcdm` | ΛCDM | TT, EE, TE, PP | 6 | ✅ Production |
| `class_mnuw0wacdm` | w₀wₐCDM + Mν | TT, EE, TE, PP | 9 | ✅ Production |
| `axiclass` | Axion Dark Matter | TT, EE, TE | 8+ | ⚙️ Development |

### Effort.jl Framework (Galaxy Power Spectra)

| Emulator | Theory Code | Redshifts | Multipoles | Status |
|----------|-------------|-----------|------------|--------|
| `pybird_mnulcdm` | PyBird | Variable | P₀, P₂, P₄ | ✅ Production |
| `pybird_mnulcdm_fixed_z` | PyBird | fixed | P₀, P₂, P₄ | ✅ Production |
| `pybird_mnuw0wacdm` | PyBird | Variable | P₀, P₂, P₄ | ✅ Production |
| `pybird_w0wacdm` | PyBird | Variable | P₀, P₂, P₄ | ✅ Production |
| `velocileptors_rept_lcdm` | Velocileptors REPT | Variable | P₀, P₂ | ⚙️ Development |
| `velocileptors_rept_mnuw0wacdm` | Velocileptors REPT | Variable | P₀, P₂, P₄ | ⚙️ Development |
| `velocileptors_rept_mnuOkcdm` | Velocileptors REPT | Variable | P₀, P₂, P₄ | ⚙️ Development|
| `velocileptors_lpt_mnuw0wacdm` | Velocileptors LPT | Variable | P₀, P₂, P₄ | ⚙️ Development |
| `velocileptors_lpt_mnuOkcdm` | Velocileptors LPT | Variable | P₀, P₂, P₄ | ⚙️ Development|

## 📊 Usage Examples

### Basic CMB Power Spectrum Prediction

```julia
using Capse

# Load a trained emulator
emulator = Capse.load_emulator("path/to/trained/emulator/")

# Define cosmological parameters
# [ln10As, ns, H0, ombh2, omch2, τ, Mν, w0, wa]
params = [3.05, 0.965, 67.5, 0.0224, 0.120, 0.054, 0.06, -1.0, 0.0]

# Get CMB power spectrum
Cℓ = Capse.get_Cℓ(params, emulator)
```

### Ensemble Prediction with Phalanx.jl

```julia
using Phalanx

# Load ensemble of emulators
ensemble = load_ensemble("phalanx_master_copy/TT")

# Get prediction with uncertainty
Cℓ_mean, Cℓ_std = predict_with_std(ensemble, params)
```

### Galaxy Power Spectrum Multipoles

```julia
using Effort

# Load trained emulator
emulator = Effort.load_emulator("pybird_mnulcdm/trained/")

# Parameters: cosmology + bias parameters
params = [ωb, ωc, h, ns, As, Mν, b1, b2, b3, b4, b5, b6, b7]

# Get power spectrum multipoles at specific k-values
P0, P2, P4 = Effort.get_Pk_multipoles(params, k_array, z, emulator)
```

### Running Validation

```bash
# Validate single emulator
julia --project=. Capse.jl/class_mnuw0wacdm/validator.jl

# Validate with Phalanx ensemble
julia --project=. validator_phalanx.jl

# Compare results
julia --project=. compare_validators.jl
```

## 🔧 Advanced Features

### Training New Emulators

```julia
# 1. Generate training data
julia --project=. data_generation.jl --samples 100000 --cosmology mnuw0wacdm

# 2. Train the emulator
julia --project=. trainer.jl --config nn_setup.json --epochs 5000

# 3. Validate performance
julia --project=. validator.jl --test_set validation_data/
```

### Parallel Data Generation

For large-scale training data generation:

```bash
# Submit to SLURM cluster
sbatch submission_job.sh

# Or run locally with multiple workers
julia -p 8 data_generation.jl --parallel
```

## 📁 Repository Structure

```
emulator-zoo/
├── Capse.jl/                    # CMB emulators
│   ├── class_lcdm/              # Standard ΛCDM
│   ├── class_mnuw0wacdm/        # Extended cosmology
│   └── axiclass/                # Axion models
├── Effort.jl/                   # LSS emulators
│   ├── pybird_*/                # PyBird-based
│   └── velocileptors_*/         # Velocileptors-based
└── validation_scripts/          # Validation tools
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- 🆕 New cosmological models
- 🎯 Improved neural architectures
- 📊 Additional validation metrics
- 🔧 Performance optimizations
- 📚 Documentation improvements

## 📬 Contact

- **Lead Developer**: Marco Bonici
- **Organization**: [CosmologicalEmulators](https://github.com/CosmologicalEmulators)
- **Issues**: [GitHub Issues](https://github.com/CosmologicalEmulators/emulator-zoo/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
