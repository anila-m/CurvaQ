# QML-Toolbox
A toolbox that analyses loss landscapes by combining metrics, ansatz characteristics and ZX-calculus

## Dependencies
The code contained in this repository requires the following dependencies:
- matplotlib==3.5.2
- networkx==2.8.8
- numpy==1.24.1
- orqviz==0.5.0
- PennyLane==0.27.0
- scipy==1.13.1
- torch==2.2.0
- cirq==0.13.1
- pyquil==3.0.1
- qiskit==1.2.4
- qiskit-aer==0.14.1
- tensorflow==2.15.0
- Flask==2.1.2
- flask-smorest==0.39.0
- ply==3.11
- qiskit_qasm3_import
- marshmallow~=3.23.1
- sympy~=1.13.3
- tqdm~=4.67.0

Install dependencies using ``pip install -r requirements.txt``  
Python 3.9.13 is the version compatible with the dependencies.

## Docker
Steps to be performed to run the application using docker.

> Prerequisites: Installed docker and docker compose.
  On Windows you can easily install Docker Desktop, which includes both.

Start the corresponding container by executing:
```
docker compose up
```

If changes were made, execute:
```
docker compose up --build
```

To forward port 8000 of the container to port 8000 of your host computer, run:
```
docker run -p 8000:8000 qmltoolbox
```

The OpenAPI documentation can be accessed at ``http://localhost:8000/docs``

## ZX-calculus
### Building

* Install Rust from https://www.rust-lang.org/tools/install
* Build the tool using `cargo build --release`

### Usage

The binary will be availabe at `target/release/bpdetect` and accepts the following arguments: `bpdetect circuitName numQubits numLayers pauliString parameterIdx`:

* `circuitName` must be one of: `introExample`, `iqpExample`, `sim1`, `sim2`, `sim9`, `sim10`, `sim11`, `sim12`, `sim15`, `iqp1`, `iqp2`, `iqp3`. Here, `introExample` and `iqpExample` are the example circuits we discuss in Sections 5.4.1 and 5.5.4 respectively.

* `pauliString` represents the measurement Hamiltonian, for example `ZXIIYX`. Should have length `numQubits`.

* `parameterIdx` is the parameter with regards to which the derivative is analysed. Counting starts at 0.

Alternatively, use the method `zx_calculus` in `main.py` and specify the desired parameters. For example:
```py
if __name__ == '__main__':
    zx_calculus(ansatz='sim1', qubits=2, layers=1, hamiltonian='ZZ', parameter=0)
```
In both cases, don't forget to build the tool first using `cargo build --release`.

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