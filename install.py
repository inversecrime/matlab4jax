import shutil
import subprocess

subprocess.run(["python", "setup.py", "bdist_wheel"])
subprocess.run(["pip", "install", "."])
shutil.rmtree("build")
shutil.rmtree("dist")
shutil.rmtree("src/matlab4jax.egg-info")
