all: src/waermeleitung.c
	mpicc -O3 -o build/waermeleitung src/waermeleitung.c

debug: src/waermeleitung.c
	mpicc -g -O0 -o build/waerme-dbg src/waermeleitung.c
