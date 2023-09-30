# $1 argument is empty, throw error
if [[ -z "$1" ]]; then
    echo "Please provide an argument: mac or linux"
    exit 1
fi

# if on mac, install this too
if [[ "$1" == "mac" ]]; then
    venv/bin/python -m pip install torch torchvision torchaudio
fi
# if on linux, install this too
if [[ "$1" == "linux" ]]; then
    venv/bin/python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# basic common install
venv/bin/python -m pip install numpy pandas matplotlib seaborn jupyter lightning transformers