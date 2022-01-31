MPICXX=mpic++
OPTIONS=-std=c++11

all: src/waermeleitung.cpp
	${MPICXX} -O3 ${OPTIONS} -o build/waermeleitung src/waermeleitung.cpp

debug: src/waermeleitung.cpp
	${MPICXX} -g -DDEBUG ${OPTIONS} -o build/waerme-dbg src/waermeleitung.cpp
