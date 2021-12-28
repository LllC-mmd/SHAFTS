# -*- makefile -*-
.PHONY: main clean test pip shaft

# OS-specific configurations
ifeq ($(OS),Windows_NT)
	PYTHON_exe = ${CONDA}/bin/python.exe
	CONDA_exe_t = ${CONDA}/condabin/conda.bat

else
	UNAME_S := $(shell uname -s)

	ifeq ($(UNAME_S),Linux) # Linux
		PYTHON_exe = ${CONDA}/bin/python
		CONDA_exe_t = ${CONDA}/condabin/conda

	endif

	ifeq ($(UNAME_S),Darwin) # macOS
		PYTHON_exe = ${CONDA}/bin/python
		CONDA_exe_t = ${CONDA}/condabin/conda

	endif

endif


PYTHON := $(if $(PYTHON_exe),$(PYTHON_exe),python)
CONDA_t := $(if $(CONDA_exe_t),$(CONDA_exe_t),conda)
# All the files which include modules used by other modules (these therefore
# need to be compiled first)

MODULE = shaft

main:
	${CONDA_t} activate
	$(MAKE) clean
	${PYTHON} setup.py bdist_wheel

# install package in dev mode and do pytest
test:
	${CONDA_t} activate
	pip install -e .
	${CONDA}/bin/pytest -s ${MODULE}/testCase

# If wanted, clean all *.o files after build
clean:
	rm -rf ${MODULE}/*.so ${MODULE}/*.pyc ${MODULE}/__pycache__ ${MODULE}/*.dSYM
	rm -rf ${MODULE}/utils/*.pyc ${MODULE}/utils/__pycache__
	rm -rf build dist
	rm -rf .eggs .pytest_cache

# upload wheels to pypi using twine
upload:
	twine upload --skip-existing dist/*whl
