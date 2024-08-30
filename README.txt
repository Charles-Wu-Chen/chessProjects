conda create -n chessproject
conda activate chessproject

(only works with CMD,   not powershell)

pip install -r requirements.txt


this is to support SVG image output.  it is a requirement for cairosvg

For Windows:
Install GTK+ and Cairo:

You can use the MSYS2 environment to install Cairo and its dependencies.
After installing MSYS2, open the MSYS2 shell and run the following commands:
bash
Copy code
pacman -S mingw-w64-x86_64-gtk3 mingw-w64-x86_64-cairo
Ensure that the MSYS2 mingw64 bin directory is added to your system's PATH.
Install CairoSVG:

You can install CairoSVG through pip:
bash
Copy code
pip install cairosvg