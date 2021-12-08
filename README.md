# Quantum-Support-Vector-Machines :rocket:

As part of the Seminar: Advanced Topics in Quantum Computing at [TUM](https://www.tum.de/), I experiment with the implementation of Quantum Support Vector Machines. 

The goal of this Quantum Machine Learning Project was to successfully implement a classical machine learning algorithm, the Support Vector Machine, a quantum algorithm, Quantum Kernel Estimation and to integrate them into a Quantum Machine Learning Algorithm, Quantum Support Vector Classifier. 

The project is written in [Python](https://www.python.org/) and uses the [Qiskit](https://www.qiskit.org/) library as the quantum circuit simulator for the implementation of the quantum kernel estimation. The quadratic programming problem of the classical algorithm is implemented uing the [cvxopt](https://www.cvxopt.org/) solver. 

The source code is located in the package folder :file_cabinet: [`quantum_svm`](https://github.com/mforstenhaeusler/Quantum-Support-Vector-Machines/tree/main/quantum_svm). 
The notebooks display some benchmarking against commoly used libraries such as [scikit-learn](https://scikit-learn.org/stable/) for Support Vector Classification and [Qiskit](https://www.qiskit.org/) for quantum algorithms.  