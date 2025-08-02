# Cadi-AI

Kara Agro's flagship model for crop disease detection and mitigation.

## Table of Contents

- [About the Project](#about-the-project)
- [Directory Structure](#directory-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Training](#training)
  - [Dataset Utilities](#dataset-utilities)
- [Contributing](#contributing)
- [License](#license)

## About the Project

This repository contains the code for Cadi-AI, a YOLO-based object detection model designed to identify and classify crop diseases and pests. The project is structured to be easily configurable and extensible, with a focus on production-ready training and deployment.

## Directory Structure

```
cadi-ai/
├── .git/
├── .gitignore
├── cadi_ai_2508.ipynb
├── config.yaml
├── dataset_utils.py
├── README.md
└── train.py
```

- **`cadi_ai_2508.ipynb`**: A Jupyter notebook for experimentation, analysis, and visualization.
- **`config.yaml`**: The main configuration file for training parameters, model selection, and data paths.
- **`dataset_utils.py`**: A collection of helper functions for creating `data.yaml` files, validating datasets, and preparing data for training.
- **`train.py`**: The main training script. It uses the `config.yaml` file to configure and run the training process.
- **`README.md`**: This file.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch
- ultralytics
- PyYAML

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/minoHealth/cadi-ai.git
   ```
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```
   *(Note: A `requirements.txt` file is not yet present in the repository, but it is recommended to create one.)*

## Usage

### Configuration

The training process is controlled by the `config.yaml` file. Here you can specify the model, dataset, training parameters, and output settings.

**Example `config.yaml`:**

```yaml
# Model Configuration
model: "yolo11m.pt"

# Data Configuration
data: "data.yaml"

# Training Parameters
epochs: 100
imgsz: 640
batch: "auto"
device: 0
workers: 8
```

### Training

To start training, run the `train.py` script with the path to your configuration file:

```sh
python train.py --config config.yaml
```

The script will automatically find the optimal batch size if `batch` is set to `"auto"`.

### Dataset Utilities

The `dataset_utils.py` script provides several command-line tools for managing your dataset:

- **Create `data.yaml`:**
  ```sh
  python dataset_utils.py --create-yaml /path/to/your/dataset
  ```
- **Validate a dataset:**
  ```sh
  python dataset_utils.py --validate /path/to/your/data.yaml
  ```
- **Set up a Kaggle dataset:**
  ```sh
  python dataset_utils.py --setup-kaggle /path/to/kaggle/dataset /path/to/working/dir
  ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.