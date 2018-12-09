PATH="`pwd`/venv/bin:$PATH"
if [ ! -d venv ]; then
    virtualenv -p /usr/bin/python3.5 venv
fi

source venv/bin/activate

echo "Installing requirements"

pip install -r requirements.txt
