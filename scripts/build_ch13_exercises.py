# scripts/build_ch13_exercises.py
"""
Build and assemble Chapter 13 Exercises Notebook (13/13_Exercises.ipynb)
Integrates Part 1 to Part 6 covering Exercises 13.1 - 13.34
"""

import os
import sys
import nbformat as nbf

sys.path.insert(0, os.path.abspath("."))

from scripts.build_ch13_part1 import build_part1
from scripts.build_ch13_part2 import build_part2
from scripts.build_ch13_part3 import build_part3
from scripts.build_ch13_part4 import build_part4
from scripts.build_ch13_part5 import build_part5
from scripts.build_ch13_part6 import build_part6

def main():
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    }
    nb.metadata['language_info'] = {
        'name': 'python',
        'version': '3.10'
    }

    all_cells = []
    print("Building Part 1 (Ex 13.1 - 13.4)...")
    all_cells.extend(build_part1())
    
    print("Building Part 2 (Ex 13.5 - 13.10)...")
    all_cells.extend(build_part2())
    
    print("Building Part 3 (Ex 13.11 - 13.15)...")
    all_cells.extend(build_part3())
    
    print("Building Part 4 (Ex 13.16 - 13.20)...")
    all_cells.extend(build_part4())
    
    print("Building Part 5 (Ex 13.21 - 13.27)...")
    all_cells.extend(build_part5())
    
    print("Building Part 6 (Ex 13.28 - 13.34)...")
    all_cells.extend(build_part6())

    nb.cells = all_cells
    
    out_path = os.path.abspath("13/13_Exercises.ipynb")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Successfully generated {out_path} with {len(all_cells)} cells!")

if __name__ == "__main__":
    main()
