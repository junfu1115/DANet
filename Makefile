ROOTDIR = $(CURDIR)

lint: cpplint pylint

cpplint:
				tests/lint.py encoding cpp src kernel

pylint:
				pylint --rcfile=$(ROOTDIR)/tests/pylintrc --ignore-patterns=".*\.so$$,.*\.dll$$,.*\.dylib$$" encoding --ignore=_ext
