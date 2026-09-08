"""
PRML Notebooks Integrity Test Suite
全15章（第0章〜第14章）にわたる全64冊のJupyter Notebookの構文・構造・メタデータ整合性を自動検証
"""

import unittest
import glob
import json
import os
import nbformat


class TestNotebooksIntegrity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        cls.notebook_paths = sorted(glob.glob(os.path.join(cls.repo_root, "*", "*.ipynb")))

    def test_notebook_count(self):
        """全64冊以上のノートブックが存在することを検証"""
        self.assertGreaterEqual(
            len(self.notebook_paths), 64,
            f"Expected at least 64 notebooks, but found {len(self.notebook_paths)}"
        )

    def test_all_chapters_have_notebooks_and_exercises(self):
        """第0章から第14章の全章にノートブックおよびExercisesが存在することを検証"""
        for chapter in range(15):
            ch_dir = os.path.join(self.repo_root, str(chapter))
            self.assertTrue(os.path.isdir(ch_dir), f"Directory for Chapter {chapter} not found: {ch_dir}")

            ch_notebooks = glob.glob(os.path.join(ch_dir, "*.ipynb"))
            self.assertGreater(len(ch_notebooks), 0, f"Chapter {chapter} has no notebooks")

            exercise_notebooks = [
                nb for nb in ch_notebooks
                if "exercise" in os.path.basename(nb).lower()
            ]
            self.assertGreater(
                len(exercise_notebooks), 0,
                f"Chapter {chapter} is missing an Exercises notebook (*Exercises*.ipynb)"
            )

    def test_notebooks_json_and_nbformat_validity(self):
        """全ノートブックが valid JSON かつ nbformat v4 仕様に準拠していることを検証"""
        for path in self.notebook_paths:
            rel_path = os.path.relpath(path, self.repo_root)
            with self.subTest(notebook=rel_path):
                # 1. 有効な JSON であること
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        self.fail(f"Invalid JSON in {rel_path}: {e}")

                # 2. nbformat v4 としてロードできること
                with open(path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)

                # 3. cells リストが存在し、空でないこと
                self.assertIn("cells", nb, f"'cells' missing in {rel_path}")
                self.assertGreater(len(nb.cells), 0, f"Empty notebook found in {rel_path}")

                # 4. 最初の markdown セルに見出しが存在すること
                has_heading = any(
                    cell.cell_type == "markdown" and any(line.strip().startswith("#") for line in cell.source.splitlines())
                    for cell in nb.cells
                )
                self.assertTrue(has_heading, f"Notebook {rel_path} lacks a markdown heading (#)")

    def test_result_directories_exist(self):
        """全15章に result/ ディレクトリが存在し、可視化画像が保存されていることを検証"""
        for chapter in range(15):
            res_dir = os.path.join(self.repo_root, str(chapter), "result")
            self.assertTrue(os.path.isdir(res_dir), f"result/ directory missing in Chapter {chapter}")
            images = glob.glob(os.path.join(res_dir, "*.png")) + glob.glob(os.path.join(res_dir, "*.jpg"))
            self.assertGreater(len(images), 0, f"No generated figures in Chapter {chapter}/result")

    def test_notebooks_all_cells_executed_without_errors(self):
        """全64冊のノートブックの全コードセルが実行済みであり、エラー出力がないことを検証"""
        for path in self.notebook_paths:
            rel_path = os.path.relpath(path, self.repo_root)
            with self.subTest(notebook=rel_path):
                with open(path, "r", encoding="utf-8") as f:
                    nb = nbformat.read(f, as_version=4)

                code_cells = [c for c in nb.cells if c.cell_type == "code"]
                for idx, cell in enumerate(code_cells):
                    source = cell.source.strip()
                    if not source:
                        continue
                    # 1. 実行出力が存在すること
                    self.assertGreater(
                        len(cell.outputs), 0,
                        f"Unexecuted code cell found in {rel_path} (code cell index {idx})"
                    )
                    # 2. エラー出力がないこと
                    for out in cell.outputs:
                        if out.get("output_type") == "error":
                            ename = out.get("ename", "UnknownError")
                            evalue = out.get("evalue", "")
                            self.fail(
                                f"Execution error in {rel_path} (cell {idx}): {ename}: {evalue}"
                            )


if __name__ == "__main__":
    unittest.main()

