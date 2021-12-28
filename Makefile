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

src_dir = src


PYTHON := $(if $(PYTHON_exe),$(PYTHON_exe),python)
CONDA_t := $(if $(CONDA_exe_t),$(CONDA_exe_t),conda)
# All the files which include modules used by other modules (these therefore
# need to be compiled first)

MODULE = shaft

# default make options
main:
	${CONDA_t} activate
	$(MAKE) -C $(src_dir) main

# build wheel
wheel:
	${CONDA_t} activate
	$(MAKE) -C $(src_dir) main

# house cleaning
clean:
	$(MAKE) -C $(src_dir) clean

# make shaft and run test cases
test:
	${CONDA_t} activate
	$(MAKE) -C $(src_dir) test

# upload wheels to pypi using twine
upload:
	$(MAKE) -C $(src_dir) upload