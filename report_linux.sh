echo "===== SYSTEM REPORT ====="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "OS & Kernel:"
lsb_release -a 2>/dev/null || cat /etc/os-release
uname -a

echo -e "\n===== CPU ====="
lscpu | grep -E 'Model name|Socket|Thread|Core|CPU\(s\)'

echo -e "\n===== MEMORY ====="
free -h
grep MemTotal /proc/meminfo

echo -e "\n===== GPU ====="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected"

echo -e "\n===== STORAGE ====="
df -hT | grep -E '^/dev/'

echo -e "\n===== PYTHON / CUDA ====="
python3 --version
which python3
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)" 2>/dev/null || echo "PyTorch not found"

echo -e "\n===== NETWORK ====="
ip -brief addr show | grep -v 'LOOPBACK'

echo -e "\n===== USERS ====="
who

echo -e "\n===== END OF REPORT ====="
