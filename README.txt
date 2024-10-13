conda create -n chessproject
conda activate chessproject

(only works with CMD,   not powershell)

pip install -r requirements.txt


# This is to support SVG image output. It is a requirement for cairosvg

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

# Running tests

To run the tests for the sharpnessLC0 function:

1. Ensure that you have the `functions.py` and `test_sharpness_lc0.py` files in the same directory.
2. Open a command prompt or terminal.
3. Navigate to the directory containing these files.
4. Run the following command:

   python -m unittest test_sharpness_lc0.py

This will execute the test cases and display the results in the console.