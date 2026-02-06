# Capstone MineDojo Setup

This README captures the working local environment setup for MineDojo on macOS with Python 3.9.

## Create Python 3.9 venv (macOS)

```bash
# Install pyenv if needed
brew install pyenv

# Install Python 3.9 and create venv
pyenv install 3.9.25
~/.pyenv/versions/3.9.25/bin/python -m venv ./.venv
```

## Install JDK 8 (macOS)

```bash
softwareupdate --install-rosetta --agree-to-license
brew install --cask temurin@8
```

Set `JAVA_HOME` when activating the venv:
```bash
export JAVA_HOME=/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home
```

## Install (from Capstone root)

```bash
# Use the Python 3.9 venv at ./.venv
./.venv/bin/python -m pip install "pip<23" "setuptools<65" "wheel<0.40" "packaging<23"
./.venv/bin/python -m pip install -r ./requirements.txt
./.venv/bin/python -m pip install -e ./MineDojo
```

## Apply Gradle Patch

```bash
git -C ./MineDojo apply ./gradle-fix.patch
```

## Validate

```bash
./.venv/bin/python ./MineDojo/scripts/validate_install.py
```

## Notes

- `numpy<2` is required because MineDojo currently references `np.unicode_`, which was removed in NumPy 2.0.
- `opencv-python==4.8.1.78` is pinned to stay compatible with `numpy<2` on Python 3.9.
- `gym==0.21.0` can fail to build with newer packaging tooling. Fix with:
```bash
./.venv/bin/python -m pip install "pip<23" "setuptools<65" "wheel<0.40" "packaging<23"
```
- Malmo build fix (local patch): update `minedojo/sim/Malmo/Minecraft/build.gradle` to use the SpongePowered repo and `org.spongepowered:mixingradle:0.6-SNAPSHOT` instead of the missing JitPack SHA.
