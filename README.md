# Deep Learning for Cardiac Arrhythmia Detection and Real-Time Prediction of Sepsis Shock

This project aims to develop deep learning models for two important applications in healthcare: cardiac arrhythmia detection from ECG data and real-time prediction of sepsis shock from ECG data.

<!-- Project Components -->
## Project Components
The project includes the following main components:

* data: this directory includes the MIT-BIH Arrhythmia Database for cardiac arrhythmia detection and the MIMIC-III dataset for sepsis shock prediction. It also includes scripts for data pre-processing and feature extraction.
pipeline: this directory includes the pipeline for training and testing the deep learning models. It consists of several subdirectories:
* modeling: includes scripts for building and training the deep learning models for both applications.
* evaluation: includes scripts for evaluating the performance of the models on validation and test sets.
* prediction: includes scripts for real-time prediction of sepsis shock using the trained models.
* visualization: includes scripts for visualizing the ECG signals and the predictions of the models.
* tests: includes unit tests and integration tests for the pipeline.
* docs: includes the documentation for the project, including the project proposal, the final report, and the technical documentation.


<!-- Usage -->
## Usage
To use the pipeline, follow these steps:

* Clone the repository:
```sh
git clone https://github.com/username/deep-learning-healthcare.git
```

* Install the required dependencies:
```sh
pip install -r requirements.txt
```

* Run the scripts in the pipeline directory to pre-process the data, train and evaluate the models, and visualize the results. For example, to train the deep learning model for cardiac arrhythmia detection using the MIT-BIH Arrhythmia Database, run the following command:
```sh
python pipeline/modeling/train_ecg_classification.py --data_path data/mit-bih-arrhythmia-database-1.0.0/
```



<!-- Documentation -->
## Documentation
The documentation for the project can be found in the docs directory. It includes the following documents:

* proposal.pdf: the original project proposal.
* final_report.pdf: the final report of the project, including the methodology, results, and conclusions.
* technical_documentation.md: technical documentation for the pipeline, including the directory structure, data pre-processing, feature extraction, model architecture, training, and evaluation.
* README.md: this README file.



<!-- Tests -->
## Tests
To run the tests for the pipeline, run the following command:

```sh
pytest
```

<!-- License -->
## License
This project is licensed under the MIT License. See the LICENSE file for details.
