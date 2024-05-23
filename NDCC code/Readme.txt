This code was created in by Ms. Mahsa Azarshab (PhD student) and Dr. Charalampos Chelmis, Associate Professor in Computer Science, both  members of the Intelligent Big Data Analytics, Applications, and Systems (IDIAS) Lab in Albany, New York, USA. For questions regarding this code, or experimental results related to it, please contact Dr. Chelmis at cchelmis@albany.edu.

Steps to Execute NDCC:
Follow these steps to correctly execute the NDCC project components in sequence, ensuring that the environment is prepared for each phase:

1- Preprocess Data: Open and run Preprocess.ipynb to prepare the datasets needed for training and validation.
2- Train Model: Execute Pre_model_feed_forward.ipynb to train the pre-trained model (g).
3- Main Algorithm: Run Run.ipynb to execute the main algorithm.
4- Validate Model: Execute Validation.ipynb to assess the accuracy of the trained model on the validation dataset and to save the predictions and probabilities.

Implementation Environment:
Our code is implemented in Python 3.8.17 with PyTorch 1.12.0, hosted on Oracle Cloud Infrastructure (OCI) using its Data Science and Data Flow services. The environment is equipped with 18 cores, 288 GB of memory, and runs on the Oracle Linux operating system. We used OCI for our experiments but have removed relevant variables to improve the interoperability of the code.

Specific Dependencies and Versions:
Oracle Cloud Infrastructure (OCI): oci 2.112.0
NumPy: numpy 1.26.4
Pandas: pandas 1.5.3
PyTorch: torch 1.12.0
IPython Kernel: ipykernel 6.15.2
Notebook: notebook 6.5.4

Note:
Please ensure that all dependencies are properly installed before running the project notebooks. For running the notebooks, consider installing JupyterLab if you need an interactive environment, which can be installed via pip install jupyterlab.
