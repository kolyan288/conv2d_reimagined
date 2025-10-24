sudo apt-get install python3-pip -y

# sudo apt install python3.10-venv -y
# curl -sSL https://pdm-project.org/install-pdm.py | python3 -
# export PATH=/home/appuser/.local/bin:$PATH
# sudo apt-get install x11-apps -y
if ( nvidia-smi ) < /dev/null > /dev/null 2>&1; then
        echo "================= GPU"
        wget -nc https://download.pytorch.org/whl/cu121/torch-2.1.2%2Bcu121-cp310-cp310-linux_x86_64.whl
        pip3 install torch-2.1.2+cu121-cp310-cp310-linux_x86_64.whl  --no-dependencies
        pip3 install typing-extensions==4.8.0
        pip3 install sympy
        pip3 install numpy==1.26.0
else
        echo "================= CPU"
        wget -nc https://download.pytorch.org/whl/cpu/torch-2.1.2%2Bcpu-cp310-cp310-linux_x86_64.whl
        pip3 install torch-2.1.2+cpu-cp310-cp310-linux_x86_64.whl
        pip3 uninstall numpy -y
        pip3 install numpy==1.26.0
fi

pip install onnx onnxsim 
# pip install tensorrt









