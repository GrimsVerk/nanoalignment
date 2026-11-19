"""Single-action tasks, graded by EXECUTING the model's answer in the sandbox.

Same 20 tasks for bash and Python. Each task runs in a fresh copy of FIXTURE.
Grading: compare stdout (normalized) and/or the final files.
"""

FIXTURE = {
    "app.log": "\n".join([
        "2026-01-01 10:00:01 INFO server started",
        "2026-01-01 10:00:05 INFO user alice logged in",
        "2026-01-01 10:01:10 WARN disk usage 81%",
        "2026-01-01 10:02:00 ERROR failed to open config.yaml",
        "2026-01-01 10:02:30 INFO user bob logged in",
        "2026-01-01 10:03:45 ERROR timeout talking to db",
        "2026-01-01 10:04:00 INFO cache warmed",
        "2026-01-01 10:05:12 WARN slow request /api/items",
        "2026-01-01 10:06:00 ERROR worker 3 crashed",
        "2026-01-01 10:07:00 INFO shutdown requested",
    ]) + "\n",
    "data.csv": "name,dept,salary\nalice,eng,120\nbob,sales,90\ncarol,eng,135\ndave,ops,80\nerin,sales,95\n",
    "numbers.txt": "17\n4\n42\n8\n23\n15\n",
    "words.txt": "apple\nfoo bar\nbanana\nfoo\ncherry\napple\nzebra foo\n",
    "notes/a.txt": "first note\n",
    "notes/b.txt": "second note\n",
    "notes/c.md": "# markdown note\n",
    "src/main.py": "import util\n\nprint(util.greet('x'))\n",
    "src/util.py": "def greet(n):\n    # TODO: handle empty names\n    return 'hi ' + n\n",
    "src/old.py": "# TODO remove this file\nOLD = True\n",
}

# few-shot examples: (task, bash, python). Not among the test tasks.
FEWSHOT = [
    ("Print the number of files in the notes directory.",
     "ls notes | wc -l",
     "import os\nprint(len(os.listdir('notes')))"),
    ("Print the lines of words.txt that contain 'an'.",
     "grep 'an' words.txt",
     "for line in open('words.txt'):\n    if 'an' in line:\n        print(line, end='')"),
    ("Create an empty file named done.txt.",
     "touch done.txt",
     "open('done.txt', 'w').close()"),
    ("Print the third line of app.log.",
     "sed -n '3p' app.log",
     "print(open('app.log').read().splitlines()[2])"),
]

# check kinds: out = exact stdout (stripped lines), outset = same set of lines,
# num = single number, files = predicate on final files (dict relpath->text)
TASKS = [
    dict(id=1, task="Print the number of lines in app.log.", check="num", want=10,
         bash="wc -l < app.log", py="print(len(open('app.log').readlines()))"),
    dict(id=2, task="Print every line of app.log that contains ERROR.", check="out",
         bash="grep ERROR app.log", py="for l in open('app.log'):\n    if 'ERROR' in l: print(l, end='')"),
    dict(id=3, task="Print how many lines of app.log contain ERROR.", check="num", want=3,
         bash="grep -c ERROR app.log", py="print(sum('ERROR' in l for l in open('app.log')))"),
    dict(id=4, task="Print the sum of the numbers in numbers.txt (one number per line).", check="num", want=109,
         bash="awk '{s+=$1} END {print s}' numbers.txt", py="print(sum(int(x) for x in open('numbers.txt')))"),
    dict(id=5, task="Print the largest number in numbers.txt.", check="num", want=42,
         bash="sort -n numbers.txt | tail -1", py="print(max(int(x) for x in open('numbers.txt')))"),
    dict(id=6, task="Print the names of the .txt files in the notes directory.", check="outset_base",
         bash="ls notes/*.txt", py="import glob\nfor f in glob.glob('notes/*.txt'): print(f)"),
    dict(id=7, task="Print the dept column of data.csv for every row except the header.", check="out",
         bash="tail -n +2 data.csv | cut -d, -f2", py="for l in open('data.csv').read().splitlines()[1:]: print(l.split(',')[1])"),
    dict(id=8, task="Print the number of different departments in data.csv (skip the header).", check="num", want=3,
         bash="tail -n +2 data.csv | cut -d, -f2 | sort -u | wc -l",
         py="print(len({l.split(',')[1] for l in open('data.csv').read().splitlines()[1:]}))"),
    dict(id=9, task="Print the names of the files in src that contain the text TODO.", check="outset_base",
         bash="grep -l TODO src/*", py="import glob\nfor f in glob.glob('src/*'):\n    if 'TODO' in open(f).read(): print(f)"),
    dict(id=10, task="Create the directory out/reports, including missing parent directories.", check="files",
         bash="mkdir -p out/reports", py="import os\nos.makedirs('out/reports', exist_ok=True)"),
    dict(id=11, task="Create a directory named backup and copy data.csv into it.", check="files",
         bash="mkdir backup && cp data.csv backup/", py="import os, shutil\nos.makedirs('backup', exist_ok=True)\nshutil.copy('data.csv', 'backup/')"),
    dict(id=12, task="Replace every occurrence of foo with bar in words.txt, editing the file in place.", check="files",
         bash="sed -i 's/foo/bar/g' words.txt", py="t = open('words.txt').read()\nopen('words.txt', 'w').write(t.replace('foo', 'bar'))"),
    dict(id=13, task="Print the first 3 lines of app.log.", check="out",
         bash="head -n 3 app.log", py="print(''.join(open('app.log').readlines()[:3]), end='')"),
    dict(id=14, task="Print the last line of app.log.", check="out",
         bash="tail -n 1 app.log", py="print(open('app.log').read().splitlines()[-1])"),
    dict(id=15, task="Print the number of words in words.txt.", check="num", want=9,
         bash="wc -w < words.txt", py="print(len(open('words.txt').read().split()))"),
    dict(id=16, task="Print the lines of words.txt sorted alphabetically with duplicates removed.", check="out",
         bash="sort -u words.txt", py="for l in sorted(set(open('words.txt').read().splitlines())): print(l)"),
    dict(id=17, task="Print the total of the salary column in data.csv.", check="num", want=520,
         bash="tail -n +2 data.csv | awk -F, '{s+=$3} END {print s}'",
         py="print(sum(int(l.split(',')[2]) for l in open('data.csv').read().splitlines()[1:]))"),
    dict(id=18, task="Delete all .md files in the notes directory.", check="files",
         bash="rm notes/*.md", py="import glob, os\nfor f in glob.glob('notes/*.md'): os.remove(f)"),
    dict(id=19, task="Create a file hello.txt that contains the line hello world.", check="files",
         bash="echo 'hello world' > hello.txt", py="open('hello.txt', 'w').write('hello world\\n')"),
    dict(id=20, task="Print the name of the person with the highest salary in data.csv.", check="out",
         bash="tail -n +2 data.csv | sort -t, -k3 -n | tail -1 | cut -d, -f1",
         py="rows = [l.split(',') for l in open('data.csv').read().splitlines()[1:]]\nprint(max(rows, key=lambda r: int(r[2]))[0])"),
]


def files_ok(tid, files):
    """Final-state checks for file tasks. `files` maps relpath -> text."""
    keep = {k: v for k, v in FIXTURE.items()}
    if tid == 10:
        return any(k.startswith("out/reports") for k in files) or files.get("__dirs__", "").find("out/reports") >= 0
    if tid == 11:
        return files.get("backup/data.csv") == FIXTURE["data.csv"] and files.get("data.csv") == FIXTURE["data.csv"]
    if tid == 12:
        return files.get("words.txt") == FIXTURE["words.txt"].replace("foo", "bar")
    if tid == 18:
        return ("notes/c.md" not in files and "notes/a.txt" in files and "notes/b.txt" in files
                and "data.csv" in files)
    if tid == 19:
        return files.get("hello.txt", "").strip() == "hello world"
    raise KeyError(tid)
