"""Runs every test_* without pytest: python herd/tests/run_all.py"""
import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
failed = 0
for f in sorted(os.listdir(HERE)):
    if f.startswith("test_") and f.endswith(".py"):
        mod = importlib.import_module(f[:-3])
        for name in dir(mod):
            if name.startswith("test_"):
                try:
                    getattr(mod, name)()
                    print(f"ok    {f}::{name}")
                except Exception:
                    failed += 1
                    print(f"FAIL  {f}::{name}")
                    traceback.print_exc()
print("all passed" if not failed else f"{failed} failed")
sys.exit(1 if failed else 0)
