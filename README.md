# CurvaQ: Total Absolute Scalar Curvature for Cost Landscapes
Experiment/Code for reproduction of results for Bachelor Thesis "Integrating Curavture into the Cost Landscape of Variational Quantum Algorithms" by Alina Mürwald (2025). The code is adapted from a student research project "QML Toolbox" (2024).

It implements Total Absolute Scalar Curvature (TASC) and contains an experiment with 3D and 6D QNN cost landscapes.

## Directory

Files that start with `BA_` were made for this project. All other files were already part of  "QML Toolbox".

TASC is implemented using Naive Monte Carlo (MC) Integration in `metrics.py`, all TASC associated functions are indicated by `#BA CurvaQ` in that file.

The implementation of MISER (Variant of MC Integration) was taken from here: https://github.com/karoliina/MISER

The main experiment method is contained in `BA_main_experiment.py`. 

Results of the final experiment (used in thesis) can be found in results/main_experiment.

Results for the preliminary tests regarding different MC Integration variants (Naive and MISER) and numbers of samples points can be found in results/preliminary_tests.

All plots can be found in the folder `plots`.


## Dependencies
The code contained in this repository requires the following dependencies:
- matplotlib==3.8.0
- networkx==2.8.8
- numpy==1.26.4
- orqviz==0.6.0
- PennyLane==0.27.0
- scipy==1.15.1
- torch==2.3.1
- cirq==0.13.1
- pyquil==3.0.1
- qiskit==1.2.4
- qiskit-aer==0.14.1
- tensorflow==2.18.0
- Flask==3.1.0
- flask-smorest==0.45.0
- ply==3.11
- qiskit_qasm3_import==0.5.1
- marshmallow~=3.23.1
- sympy~=1.13.3
- tqdm~=4.67.0

Install dependencies using ``pip install -r requirements.txt``.  

The experiments were run on Python 3.11.7.

## Disclaimer of Warranty
Unless required by applicable law or agreed to in writing, 
Licensor provides the Work (and each Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation,
any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.
You are solely responsible for determining the appropriateness of using or redistributing the Work
and assume any risks associated with Your exercise of permissions under this License.

## Haftungsausschluss
Dies ist ein Forschungsprototyp.
Die Haftung für entgangenen Gewinn, Produktionsausfall, Betriebsunterbrechung, entgangene Nutzungen,
Verlust von Daten und Informationen, Finanzierungsaufwendungen sowie sonstige Vermögens- und Folgeschäden ist,
außer in Fällen von grober Fahrlässigkeit, Vorsatz und Personenschäden, ausgeschlossen.
