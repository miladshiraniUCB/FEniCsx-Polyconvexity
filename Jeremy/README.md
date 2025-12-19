Selecting Python interpreter from WSL (For windows only):
Install the remote WSL extension
Then, on the bottom left of VS code, click the >< button and select "Connect to WSL"

Download python3 to WSL:
sudo apt update
sudo apt install python3 python3-pip

Create a virtual environment (venv)
sudo apt install python3-venv
python3 -m venv .venv

To activate the virtual environment
source .venv/bin/activate

To install FENICSx via conda
https://me.jhu.edu/nguyenlab/doku.php?id=fenicsx

Activate FENICSx once conda is installed
conda activate fenicsx
