# ConvLSTM Satellite Imagery Analysis

This project implements a ConvLSTM (Convolutional Long Short-Term Memory) network for analyzing satellite imagery. The ConvLSTM architecture is designed to capture spatial and temporal dependencies in sequences of images, making it suitable for tasks such as change detection, land cover classification, and other applications in remote sensing.

## Project Structure

```
conv_lstm_satellite
├── src
│   ├── dataset
│   │   └── satellite_dataset.py  # Custom dataset class for loading satellite images
│   ├── models
│   │   ├── convlstm_cell.py       # Definition of the ConvLSTM cell
│   │   └── convlstm_network.py     # Overall architecture of the ConvLSTM network
│   ├── train.py                    # Training routine for the ConvLSTM network
│   └── utils
│       └── __init__.py            # Initialization file for the utils module
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── config.yaml                     # Configuration file for training parameters
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Install Dependencies**: 
   Navigate to the project directory and install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**: 
   Place your satellite imagery in a directory structure that the `SatelliteDataset` class can access. Ensure that the images are in a format supported by PyTorch.

4. **Configure Training Parameters**: 
   Edit the `config.yaml` file to set your desired training parameters, such as learning rate, batch size, and dataset paths.

5. **Run Training**: 
   Execute the training script to start training the ConvLSTM network:
   ```
   python src/train.py
   ```

## Usage Examples

After training, you can use the trained model for inference on new satellite imagery. Refer to the `train.py` file for examples on how to load the model and make predictions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.