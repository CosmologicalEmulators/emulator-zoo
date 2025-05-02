# emulator-zoo
![Image](https://github.com/user-attachments/assets/60c59b07-e758-44f9-b03a-925b6b70c155)

Please do not feed the models

## Emulators for Cosmological Observables

[![License](https://img.shields.io/github/license/CosmologicalEmulators/emulator-zoo)](LICENSE)
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Data Handling](#data-handling)
    - [Downloading from Zenodo](#downloading-from-zenodo)
    - [Creating Datasets Locally](#creating-datasets-locally)
- [Usage](#usage)
    - [Training an Emulator](#training-an-emulator)
    - [Validating an Emulator](#validating-an-emulator)
    - [Using a Trained Emulator](#using-a-trained-emulator)
- [Emulator Zoo Structure](#emulator-zoo-structure)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

## Description

Welcome to the `emulator-zoo`, a repository within the [CosmologicalEmulators](https://github.com/CosmologicalEmulators) organization. This repository serves as a centralized collection of scripts and code necessary to generate training data, train, validate, and utilize emulators for various cosmological observables, such as the Cosmic Microwave Background (CMB) power spectra and galaxy power spectrum multipoles.

Our goal is to provide a reproducible framework for developing and deploying high-accuracy cosmological emulators. By housing the training and validation code alongside the necessary configurations and **scripts for both downloading existing datasets from Zenodo and generating custom datasets locally**, we ensure that the emulators can be retrained, verified, and adapted by anyone. This repository heavily leverages the capabilities of the [`EmulatorsTrainer.jl`](https://github.com/CosmologicalEmulators/EmulatorsTrainer.jl) Julia package for streamlined emulator development workflows.

## Features

* **Data Generation Scripts:** Includes scripts to generate training datasets from simulations or theoretical models, allowing users to customize datasets for their specific needs.
* **Zenodo Download Scripts:** Provides scripts to easily download pre-generated training datasets hosted on Zenodo.
* **Reproducible Training:** Contains the exact scripts and configurations used to train the emulators.
* **Validation Framework:** Includes code and procedures for validating the performance and accuracy of trained emulators.
* **Diverse Observables:** Supports the emulation of key cosmological observables like CMB and galaxy power spectra.
* **`EmulatorsTrainer.jl` Integration:** Built upon the `EmulatorsTrainer.jl` package for efficient and standardized training pipelines.
* **Centralized Repository:** Provides a single location for accessing and contributing to the collection of cosmological emulators.

## Installation

To get started with the `emulator-zoo`, you will need to have Julia installed. If you don't have Julia, please follow the instructions on the [official Julia website](https://julialang.org/downloads/).

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/CosmologicalEmulators/emulator-zoo.git](https://github.com/CosmologicalEmulators/emulator-zoo.git)
    cd emulator-zoo
    ```

2.  **Instantiate the Julia environment:**
    ```julia
    julia --project=.
    (emulator-zoo) pkg> instantiate
    ```
    This will install all the necessary Julia dependencies, including `EmulatorsTrainer.jl`.

3.  **Obtain Training Data:**
    Before training emulators, you will need the required training datasets. Refer to the [Data Handling](#data-handling) section for instructions on downloading pre-built datasets or generating your own.

## Data Handling

This repository provides flexibility in obtaining the necessary training datasets. You can either download pre-generated datasets from Zenodo or use the provided scripts to create your own datasets locally.

### Downloading from Zenodo

Pre-generated training datasets for the emulators in this zoo are archived on Zenodo. You can download them using the provided scripts:

```bash
julia --project=. scripts/data_handling/download_from_zenodo.jl --dataset <dataset_name>
```
