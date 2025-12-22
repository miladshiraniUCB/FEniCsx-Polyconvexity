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
per https://me.jhu.edu/nguyenlab/doku.php?id=fenicsx
FEniCSx on Linux and macOS
As of March 2023, the latest stable release of dolfinx available via Anaconda is 0.6. If you build FEniCSx from the source, you can perhaps install 0.7.

Once Anaconda is properly installed, create an environment for FEniCSx. In addition to the dolfinx library, we will install mpich, pyvista, matplotlib, and cycler. Parallel processing library mpich allows different multi-processor operations within FEniCSx and the other three packages are used for quick visualization. Standard installation of Anaconda already comes with these three packages but we will still need to install them inside the FEniCSx environment.
(base)    $ conda create -n fenicsx
(base)    $ conda activate fenicsx
(fenicsx) $ conda install -c conda-forge fenics-dolfinx mpich pyvista matplotlib cycler 
Plain text
pyvista supports plotting higher order unstructured mesh.matplotlib lacks support for higher order unstructured mesh. So, it is recommended to use pyvista for quick visualization. But you can use matplotlib for regular plotting.
To uninstall FEniCSx packages from Anaconda (Only do this step whenever you need to uninstall/ re-install the package), you will have to uninstall everything within the environment. Before you proceed to uninstall check if FEniCSx environment is active in the terminal. If it is active, then deactivate it first and proceed to uninstall the packages.
(fenicsx) $ conda deactivate
(base)    $ conda remove -n fenicsx --all
(base)    $ conda clean --all 
Plain text
It will ask your permission; proceed as needed. FEniCSx should be completely uninstalled now.

Activate FENICSx once conda is installed
conda activate fenicsx


To download anaconda (https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da):
Install WSL (Ubuntu for Windows - can be found in Windows Store). I recommend the latest version (I'm using 18.04) because there are some bugs they worked out during 14/16 (microsoft/WSL#785)
Go to https://repo.continuum.io/archive to find the list of Anaconda releases
Select the release you want. I have a 64-bit computer, so I chose the latest release ending in x86_64.sh. If I had a 32-bit computer, I'd select the x86.sh version. If you accidentally try to install the wrong one, you'll get a warning in the terminal. I chose Anaconda3-5.2.0-Linux-x86_64.sh.
From the terminal run wget https://repo.continuum.io/archive/[YOUR VERSION]. Example: $ wget https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh
Run the installation script: $ bash Anaconda[YOUR VERSION].sh ($ bash Anaconda3-5.2.0-Linux-x86_64.sh)
Read the license agreement and follow the prompts to accept. When asks you if you'd like the installer to prepend it to the path, say yes.
Optionally install VS Code when prompted (some have reported this installation doesn't work - checkout https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da#gistcomment-3665550)
Close the terminal and reopen it to reload .bash configs.
To test that it worked, run $ which python. It should print a path that has anaconda in it. Mine is /home/kauff/anaconda3/bin/python. If it doesn't have anaconda in the path, do the next step. Otherwise, move to step 11.
Manually add the Anaconda bin folder to your PATH. To do this, I added "export PATH=/home/kauff/anaconda3/bin:$PATH" to the bottom of my ~/.bashrc file.
To open jupyter, type $ jupyter notebook --no-browser. The no browser flag will still run Jupyter on port 8888, but it won't pop it open automatically. it's necessary since you don't have a browser (probably) in your subsystem. In the terminal, it will give you a link to paste into your browser. If it worked, you should see your notebooks!