# $1 argument is empty, throw error
if [[ -z "$1" ]]; then
    echo "Please provide an argument: mac or linux"
    exit 1
fi

# if on mac, install this too
if [[ "$1" == "mac" ]]; then
    pip install torch torchvision torchaudio
    exit 0
fi

# if on linux, install this too
if [[ "$1" == "linux" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# basic common install
pip install numpy pandas matplotlib seaborn jupyter lightning