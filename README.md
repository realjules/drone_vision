# VisDrone Object Detection with YOLOv8

This project implements an object detection system using YOLOv8 on the VisDrone dataset. It includes data preprocessing, model training, evaluation, and visualization components.

## Project Structure

```
project_root/
│
├── data_preprocessing.py
├── augmentation.py
├── model.py
├── evaluation.py
├── visualization.py
├── utils.py
├── config.py
├── main.py
├── requirements.txt
└── README.md
```

- `data_preprocessing.py`: Handles dataset preparation and annotation conversion.
- `augmentation.py`: Implements data augmentation techniques.
- `model.py`: Contains functions for loading, training, and running inference with the YOLOv8 model.
- `evaluation.py`: Provides metrics calculation and model evaluation utilities.
- `visualization.py`: Includes functions for visualizing results and metrics.
- `utils.py`: Contains utility functions used across the project.
- `config.py`: Stores configuration parameters and paths.
- `main.py`: The entry point of the program, orchestrating the entire pipeline.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/realjules/drone_vision.git
   cd drone_vision
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the VisDrone dataset and place it in a `dataset` folder in the project root. The structure should be:
   ```
   dataset/
   ├── train/
   │   ├── annotations/
   │   └── sequences/
   └── val/
       ├── annotations/
       └── sequences/
   ```

## Usage

1. Configure the parameters in `config.py` if needed.

2. Run the main script:
   ```
   python main.py
   ```

   This will:
   - Preprocess the dataset
   - Train the YOLOv8 model
   - Evaluate the model
   - Generate visualizations

3. Check the `output` folder for results and visualizations.

## Customization

- Modify `config.py` to change dataset paths, model parameters, or output locations.
- Adjust augmentation techniques in `augmentation.py`.
- Extend evaluation metrics or visualization options in their respective files.

## Contributing

Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VisDrone dataset: [VisDrone Project](http://aiskyeye.com/)
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)