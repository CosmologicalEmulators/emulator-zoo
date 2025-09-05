# emulator-zoo 🦁

![Image](https://github.com/user-attachments/assets/60c59b07-e758-44f9-b03a-925b6b70c155)

*A comprehensive collection of cosmological emulators for high-precision theoretical predictions*

[![License](https://img.shields.io/github/license/CosmologicalEmulators/emulator-zoo)](LICENSE)
[![Julia](https://img.shields.io/badge/Julia-1.9+-purple)](https://julialang.org)
[![Status](https://img.shields.io/badge/Status-Active-green)](https://github.com/CosmologicalEmulators/emulator-zoo)

## 🎯 Overview

Welcome to **emulator-zoo**, the centralized repository for state-of-the-art cosmological emulators within the [CosmologicalEmulators](https://github.com/CosmologicalEmulators) organization. This repository provides a unified framework for training, validating, and deploying neural network emulators for various cosmological observables.

### Key Features

- 🚀 **Production-Ready Emulators**: CMB power spectra (Cℓ) and galaxy power spectrum multipoles
- 🔬 **Multiple Cosmologies**: ΛCDM, w₀wₐCDM, massive neutrinos, and axion models
- 📊 **Validated Performance**: Comprehensive validation on 30,000+ cosmological parameter combinations
- 🔧 **Ensemble Methods**: Support for Phalanx.jl ensemble averaging with 50-75% outlier reduction
- 📦 **Two Backend Frameworks**: Capse.jl (CMB) and Effort.jl (LSS)
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

# Run a quick test (example with Capse.jl)
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
| `pybird_mnulcdm_fixed_z` | PyBird | z=0.5 | P₀, P₂, P₄ | ✅ Production |
| `pybird_mnuw0wacdm` | PyBird | Variable | P₀, P₂, P₄ | ✅ Production |
| `pybird_w0wacdm` | PyBird | Variable | P₀, P₂, P₄ | ✅ Production |
| `velocileptors_lcdm` | Velocileptors | Variable | P₀, P₂ | ⚙️ Development |

## 💻 Installation

### Prerequisites

- Julia 1.9 or higher
- Python 3.8+ (for data generation)
- 8GB+ RAM recommended
- CUDA-capable GPU (optional, for accelerated training)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/CosmologicalEmulators/emulator-zoo.git
cd emulator-zoo

# 2. Install Julia dependencies
julia --project=.
julia> using Pkg
julia> Pkg.instantiate()

# 3. Install Python dependencies (if needed for data generation)
pip install -r requirements.txt  # If available
```

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

# Relative uncertainty
rel_uncertainty = Cℓ_std ./ Cℓ_mean
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

## 📈 Validation & Performance

### Recent Validation Results (Sept 2024)

Comprehensive validation on 30,173 cosmological parameter combinations shows:

#### Single vs Ensemble Performance (Phalanx)

| Spectrum | 68% CL Improvement | 95% CL Improvement | 99% CL Improvement |
|----------|-------------------|-------------------|-------------------|
| **TT** | 21% | 46% | 76% |
| **EE** | 2% | -4% | 30% |
| **TE** | 12% | 22% | 63% |
| **PP** | 2% | 11% | 55% |

**Key Achievement**: 50-75% reduction in extreme outliers at 99% confidence level

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

### Custom Ensemble Configurations

Create a `config.json` for weighted ensemble averaging:

```json
{
  "backend": "Capse",
  "emulators": [
    {"path": "emulator_1", "weight": 0.15},
    {"path": "emulator_2", "weight": 0.10},
    {"path": "emulator_3", "weight": 0.12},
    ...
  ]
}
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
├── phalanx_master_copy/         # Ensemble configurations
│   ├── TT/                      # Temperature spectrum
│   ├── EE/                      # E-mode polarization
│   ├── TE/                      # Cross-correlation
│   └── PP/                      # Lensing potential
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

## 📝 Citation

If you use emulator-zoo in your research, please cite:

```bibtex
@software{emulator_zoo_2024,
  author = {Bonici, Marco and CosmologicalEmulators Contributors},
  title = {emulator-zoo: A Comprehensive Collection of Cosmological Emulators},
  year = {2024},
  url = {https://github.com/CosmologicalEmulators/emulator-zoo},
  version = {1.0.0}
}
```

For specific emulator frameworks:
- **Capse.jl**: [Citation details]
- **Effort.jl**: [Citation details]
- **Phalanx.jl**: [Paper in preparation]

## 📬 Contact

- **Lead Developer**: Marco Bonici
- **Organization**: [CosmologicalEmulators](https://github.com/CosmologicalEmulators)
- **Issues**: [GitHub Issues](https://github.com/CosmologicalEmulators/emulator-zoo/issues)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*"Please do not feed the models" - But feel free to train them! 🦁*