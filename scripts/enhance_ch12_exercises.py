# scripts/enhance_ch12_exercises.py
"""
Orchestration script for PRML Chapter 12 Exercises.
Assembles 12/12_Exercises.ipynb, executes all cells via ExecutePreprocessor,
and saves the evaluated notebook with outputs.
"""

import os
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

# Ensure scripts dir is on sys.path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

from build_ch12_exercises import get_ch12_all_exercises

def main():
    print("=== Assembling Chapter 12 Exercises Notebook ===")
    cells = get_ch12_all_exercises()
    print(f"Total cells assembled: {len(cells)}")

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.11.12"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }
    nb.cells = cells

    target_dir = os.path.abspath("12")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "12_Exercises.ipynb")

    print(f"Executing notebook at {target_path} ...")
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": target_dir}})
        print("Notebook execution succeeded with 0 errors!")
    except Exception as e:
        print(f"Execution failed: {e}")
        with open(target_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        raise e

    print(f"Saving executed notebook to {target_path} ...")
    with open(target_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print("Chapter 12 Exercises Notebook successfully generated and evaluated!")

if __name__ == "__main__":
    main()
