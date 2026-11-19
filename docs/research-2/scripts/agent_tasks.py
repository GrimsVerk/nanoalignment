"""Multi-step pilot: 8 simple file tasks + 2 demo tasks, identical across interfaces.

Graders are hidden from the model. Code tasks are graded by running PRISTINE
tests (inside the sandbox) against the model's files. Text tasks are graded
by reading files on the host (no execution).
"""
import json

TEST_TMPL = '''import unittest
{imports}


class T(unittest.TestCase):
{body}

if __name__ == "__main__":
    unittest.main()
'''


def tests(imports, cases):
    body = "".join(f"    def test_{i}(self):\n        {c}\n\n" for i, c in enumerate(cases))
    return TEST_TMPL.format(imports=imports, body=body)


APP_LOG = "\n".join([
    "10:00:01 INFO server started",
    "10:00:05 INFO user alice logged in",
    "10:02:00 ERROR failed to open config.yaml",
    "10:02:30 WARN slow disk",
    "10:03:45 ERROR timeout talking to db",
    "10:04:00 INFO cache warmed",
    "10:06:00 ERROR worker 3 crashed",
    "10:07:00 INFO shutdown requested",
]) + "\n"

TASKS = [
    dict(id="fix_add", kind="tests", protect=["test_calc.py"],
         task="The tests in test_calc.py fail. Fix the bug in calc.py so that all tests pass. Do not change the tests.",
         files={"calc.py": "def add(a, b):\n    return a - b\n\n\ndef mul(a, b):\n    return a * b\n",
                "test_calc.py": tests("from calc import add, mul",
                                      ["self.assertEqual(add(2, 3), 5)", "self.assertEqual(add(-1, 1), 0)",
                                       "self.assertEqual(mul(2, 3), 6)"])}),
    dict(id="fix_slug", kind="tests", protect=["test_text_utils.py"],
         task="The tests in test_text_utils.py fail. Fix the bug in text_utils.py so that all tests pass. Do not change the tests.",
         files={"text_utils.py": 'def slugify(title):\n    """Turn \'Hello World\' into \'hello-world\'."""\n    return title.strip().replace(" ", "-")\n',
                "test_text_utils.py": tests("from text_utils import slugify",
                                            ['self.assertEqual(slugify("Hello World"), "hello-world")',
                                             'self.assertEqual(slugify("  Big News "), "big-news")'])}),
    dict(id="impl_even", kind="tests", protect=["test_mathx.py"],
         task="The function is_even in mathx.py is not written yet. Write it so that the tests in test_mathx.py pass.",
         files={"mathx.py": 'def is_even(n):\n    """Return True if n is even, else False."""\n    raise NotImplementedError\n',
                "test_mathx.py": tests("from mathx import is_even",
                                       ["self.assertTrue(is_even(4))", "self.assertFalse(is_even(7))",
                                        "self.assertTrue(is_even(0))"])}),
    dict(id="errors_only", kind="text", protect=["app.log"],
         task="Write all lines of app.log that contain ERROR to a new file errors.txt, in the same order. Keep app.log unchanged.",
         files={"app.log": APP_LOG}),
    dict(id="config_flag", kind="text", protect=[],
         task="In config.json, set debug to false. Keep the other settings as they are.",
         files={"config.json": '{\n  "name": "shop",\n  "debug": true,\n  "port": 8080\n}\n'}),
    dict(id="csv_total", kind="text", protect=["sales.csv"],
         task="Write the total of the amount column in sales.csv to a new file total.txt. Write only the number.",
         files={"sales.csv": "item,amount\npen,3\nbook,12\nbag,25\ncup,7\n"}),
    dict(id="todo_files", kind="text", protect=[],
         task="Some of the .py files in this directory contain the word TODO. Write their file names to todo.txt, one per line.",
         files={"a.py": "x = 1\n", "b.py": "# TODO: add checks\ny = 2\n", "c.py": "def f():\n    return 3\n",
                "d.py": "def g():\n    pass  # TODO later\n"}),
    dict(id="bump_counter", kind="text", protect=[],
         task="The file visits.txt contains a number. Increase it by 1 and save it back to visits.txt.",
         files={"visits.txt": "41\n"}),
]


def grade_text(tid, files):
    if tid == "errors_only":
        want = [l for l in APP_LOG.splitlines() if "ERROR" in l]
        got = [l.rstrip() for l in files.get("errors.txt", "").splitlines() if l.strip()]
        return got == want
    if tid == "config_flag":
        try:
            d = json.loads(files.get("config.json", ""))
        except Exception:
            return False
        return d == {"name": "shop", "debug": False, "port": 8080}
    if tid == "csv_total":
        try:
            return float(files.get("total.txt", "").strip()) == 47
        except ValueError:
            return False
    if tid == "todo_files":
        got = sorted(l.strip().split("/")[-1] for l in files.get("todo.txt", "").splitlines() if l.strip())
        return got == ["b.py", "d.py"]
    if tid == "bump_counter":
        return files.get("visits.txt", "").strip() == "42"
    raise KeyError(tid)


# ---- demos (few-shot). Actions per interface; observations are produced by
# really executing them in the sandbox when the prompt is built.
DEMOS = [
    dict(task="The tests in test_shapes.py fail. Fix the bug in shapes.py so that all tests pass. Do not change the tests.",
         files={"shapes.py": "def square_area(side):\n    return side * 2\n",
                "test_shapes.py": tests("from shapes import square_area",
                                        ["self.assertEqual(square_area(3), 9)", "self.assertEqual(square_area(1), 1)"])},
         bash=["ls", "cat shapes.py", "run_tests", "sed -i 's/side \\* 2/side * side/' shapes.py", "run_tests",
               'submit "Fixed square_area in shapes.py: it multiplied by 2 instead of squaring. All tests pass now."'],
         python=["import os\nprint(os.listdir('.'))", "print(open('shapes.py').read())", "run_tests()",
                 "src = open('shapes.py').read()\nopen('shapes.py', 'w').write(src.replace('side * 2', 'side * side'))",
                 "run_tests()",
                 'submit("Fixed square_area in shapes.py: it multiplied by 2 instead of squaring. All tests pass now.")'],
         tools=[{"tool": "list_dir", "path": "."}, {"tool": "read_file", "path": "shapes.py"}, {"tool": "run_tests"},
                {"tool": "write_file", "path": "shapes.py", "content": "def square_area(side):\n    return side * side\n"},
                {"tool": "run_tests"},
                {"tool": "submit_report", "report": "Fixed square_area in shapes.py: it multiplied by 2 instead of squaring. All tests pass now."}]),
    dict(task="Count the lines in names.txt that start with A and write the count to a new file count.txt.",
         files={"names.txt": "Anna\nBob\nAlice\nCarl\nAdam\n"},
         bash=["cat names.txt", "grep -c '^A' names.txt > count.txt", "cat count.txt",
               'submit "Counted 3 names starting with A and wrote 3 to count.txt."'],
         python=["print(open('names.txt').read())",
                 "n = sum(1 for line in open('names.txt') if line.startswith('A'))\nopen('count.txt', 'w').write(str(n))\nprint(n)",
                 'submit("Counted 3 names starting with A and wrote 3 to count.txt.")'],
         tools=[{"tool": "read_file", "path": "names.txt"}, {"tool": "write_file", "path": "count.txt", "content": "3\n"},
                {"tool": "read_file", "path": "count.txt"},
                {"tool": "submit_report", "report": "Counted 3 names starting with A and wrote 3 to count.txt."}]),
]
