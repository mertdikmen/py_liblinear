OBJS = liblinear_python.o
OBJS+= pca_nipals.o
OBJS+= libsvm_python.o
CPP = g++

LIBLINEAR_PATH=/home/mert/my/liblinear-1.92
LIBSVM_PATH=/home/mert/my/libsvm-3.12
NUMPYFLAG= -I$(shell python2 -c 'import numpy; print numpy.get_include()')
INC = $(shell python2.7-config --include) $(NUMPYFLAG)
LIB = $(shell python2.7-config --libs)

CFLAGS = -I$(LIBLINEAR_PATH) -I$(LIBSVM_PATH)
LFLAGS = -L$(LIBLINEAR_PATH) -L$(LIBSVM_PATH)
CFLAGS+= -fPIC

LIB += -lboost_python 
OPT := \
	-O3 \
	-funroll-loops \
	-ftree-loop-distribution \
	-fmerge-all-constants \
	-ftracer \
	-fvariable-expansion-in-unroller \
	-fvpt

all:
	$(MAKE) liblinear_python.so libsvm_python.so

libsvm_python.so: libsvm_python.o
	$(CPP) $(OPT) $(LIB) $(LFLAGS) -lsvm libsvm_python.o -shared -o libsvm_python.so 

libsvm_python.o: libsvm_python.cpp 
	$(CPP) $(OPT) -c libsvm_python.cpp $(CFLAGS) $(INC) 

liblinear_python.so: liblinear_python.o
	$(CPP) $(OPT) $(LIB) $(LFLAGS) -llinear liblinear_python.o -shared -o liblinear_python.so 

liblinear_python.o: liblinear_python.cpp 
	$(CPP) $(OPT) -c liblinear_python.cpp $(CFLAGS) $(INC) 

#pca_nipals.so: pca_nipals.o
#	$(CPP) $(OPT) $(LIB) $(LFLAGS) pca_nipals.o -shared -o pca_nipals.so
#
#pca_nipals.o: pca_nipals.cpp
#	$(CPP) $(OPT) -c pca_nipals.cpp $(CFLAGS) $(INC)

clean:
	rm *.o *.so
