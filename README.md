# VizML
Project for CS8803 ML System - Visualized Programming with Block Abstraction for Neural Network

## Dependencies Notice
We have encountered issues about the Cuda support for Torch 1.4 and Torch 1.5. A requirements.txt file is under the VizML folder. 
To install the correct version of Cuda with Torch, we suggest the user check the local Cuda version. We experiment and test our system in **Torch 1.4** with **Cuda 10.1** and **Torch 1.5** with **Cuda 10.1** support. Locally, we have Cuda 10.2. 
There is an update in Torch 1.5 that the Cuda version must be declared manually, otherwise, 9.2 installed.

## Dependencies Install
Since the Torch dependencies is too big to put in github repo, manual dependencies install is required. It is better to use IDE and virtual environments in Python.
Create a virtual environment under the VizML folder. Enter VizML folder and start virtual environments, enter "pip install -r requirements.txt" in console for dependencies install. (before install, see above notice for Torch and Cuda issues)

## System Running
With virtual environment activated, run **app.py**. The system frontend should be accessible in **localhost:5000/**.