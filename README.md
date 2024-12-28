## Enhancements

### Visualization Features
1. **Integrated TensorBoard**:
   - Real-time tracking of training loss, test accuracy, and client selection distribution for each round.
   - Easily monitor training progress by running:
     ```bash
     tensorboard --logdir=./logs
     ```
   - Provides a detailed, interactive view of training metrics.

2. **Performance Comparison**:
   - Automatically saves experiment results, including accuracy and loss, in a CSV file located at `./results/experiment_results.csv`.
   - Generates comparison plots for:
     - **Accuracy Comparison**: Saved as `./save/performance_comparison_accuracy.png`.
     - **Loss Comparison**: Saved as `./save/performance_comparison_loss.png`.

3. **Data Distribution Visualization**:
   - Visualizes the distribution of data samples across clients.
     - Example output: `./save/client_data_distribution.png`.
   - For MNIST, displays class distribution for a specific client.
     - Example output: `./save/class_distribution_client0.png`.

### Logging Enhancements
- Logs selected clients and average loss for each training round to both TensorBoard and the console.
- Console output provides real-time feedback on training progress, ensuring a smooth debugging experience.

## How to Run
To run the updated `main_fed.py` with enhanced features, use the following command:
```bash
python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0






# Federated Learning [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4321561.svg)](https://doi.org/10.5281/zenodo.4321561)

This is partly the reproduction of the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)   
Only experiments on MNIST and CIFAR10 (both IID and non-IID) is produced by far.

Note: The scripts will be slow without the implementation of parallel computing. 

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP and CNN models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP and CNN is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_fed.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.

## Results
### MNIST
Results are shown in Table 1 and Table 2, with the parameters C=0.1, B=10, E=5.

Table 1. results of 10 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP|  94.57%     | 70.44%         |
| FedAVG-CNN|  96.59%     | 77.72%         |

Table 2. results of 50 epochs training with the learning rate of 0.01

| Model     | Acc. of IID | Acc. of Non-IID|
| -----     | -----       | ----           |
| FedAVG-MLP| 97.21%      | 93.03%         |
| FedAVG-CNN| 98.60%      | 93.81%         |


## Ackonwledgements
Acknowledgements give to [youkaichao](https://github.com/youkaichao).

## References
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.

## Cite As
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561


