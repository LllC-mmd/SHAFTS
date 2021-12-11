# -*- makefile -*-
.PHONY: main clean test pip shaft

# OS-specific configurations
ifeq ($(OS),Windows_NT)
	PYTHON_exe = python.exe

else
	UNAME_S := $(shell uname -s)

	ifeq ($(UNAME_S),Linux) # Linux
		PYTHON_exe=python

	endif

	ifeq ($(UNAME_S),Darwin) # macOS
		PYTHON_exe=python

	endif

endif

src_dir = src


PYTHON := $(if $(PYTHON_exe),$(PYTHON_exe),python)
# All the files which include modules used by other modules (these therefore
# need to be compiled first)

MODULE = shaft

# default make options
main:
	$(MAKE) -C $(src_dir) main

# build wheel
wheel:
	$(MAKE) -C $(src_dir) main

# house cleaning
clean:
	$(MAKE) -C $(src_dir) clean

# make shaft and run test cases
test:
	$(MAKE) -C $(src_dir) test

# upload wheels to pypi using twine
upload:
	$(MAKE) -C $(src_dir) upload