# Raw TensorFlow Workloads

This directory contains scripts and configurations for running distributed TensorFlow training workloads, specifically designed to be orchestrated from a "bastion" node (e.g., a local machine or a dedicated control container) interacting with a Kubernetes cluster.

## Overview

The core of this workload is `train_tf_ps.py`, a flexible training script that supports:
- **Data Loading**: From CSV files (for structured data) or flat image directories with JSONL labels (for regression tasks like laser spot detection).
- **Model Architecture**: Configurable Deep Neural Networks (DNN) for classification or Convolutional Neural Networks (CNN) for regression.
- **Distributed Training**: Implements `tf.distribute.ParameterServerStrategy` to scale training across multiple worker nodes and parameter servers.

## Key Files

### `train_tf_ps.py`
The main Python training script.
- **Features**:
  - **`load_csv`**: Parses CSV datasets for classification tasks.
  - **`make_image_dataset`**: Creates a `tf.data.Dataset` from a directory of images and a `clean_labels.jsonl` file.
  - **`make_parameter_server_strategy`**: Sets up the distributed training cluster (ClusterSpec) and strategy.
  - **`run_deep_training` / `run_image_training`**: Entry points for training logic, handling both single-process and distributed modes.
- **Usage**:
  ```bash
  python train_tf_ps.py --data-path /path/to/data --use-ps --worker-replicas 2 ...
  ```

### `run_tf_training_from_bastion.sh`
A shell script to automate the launch of distributed training from a bastion node.
- **Service Discovery**: Uses `kubectl` to find the LoadBalancer IPs of the worker and parameter server services running in the Kubernetes cluster.
- **Network Configuration**: Auto-detects the bastion's own routable IPv4 address (`CHIEF_ADDR`) to allow workers to connect back to the coordinator.
- **Execution**: Constructs the necessary arguments (worker addresses, PS addresses, ports) and launches `train_tf_ps.py`.

### `sfs_sensor/`
A subdirectory containing model architecture definitions (e.g., `model_architecture.py`) used by the training script.

## Distributed Training Workflow

1.  **Infrastructure Setup**: A Kubernetes cluster is assumed to be running with `tf-trainer-worker` and `tf-trainer-ps` services exposed via LoadBalancers.
2.  **Bastion Execution**: The user runs `run_tf_training_from_bastion.sh` on a machine with `kubectl` access to the cluster.
3.  **Coordination**:
    - The script resolves the IPs of the cluster nodes.
    - It starts `train_tf_ps.py` as the **Chief/Coordinator**.
    - The Coordinator connects to the remote workers and parameter servers via gRPC.
4.  **Training**: The Coordinator schedules training steps on the workers, which pull data, compute gradients, and update variables on the parameter servers.

## Environment Variables

The `run_tf_training_from_bastion.sh` script respects several environment variables for configuration:
- `DATA_PATH`: Path to the dataset (default: `/data/health.csv`).
- `OUTPUT_DIR`: Directory for saved models and logs.
- `EPOCHS`: Number of training epochs.
- `BATCH_SIZE`: Batch size per step.
- `CHIEF_PORT`: Port for the coordinator to listen on (default: `2223`).
