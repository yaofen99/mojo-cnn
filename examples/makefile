CC=/opt/intel/oneapi/2024.1/bin/icpx
CFLAGS_OMP= -I../mojo/ -std=c++11 -qopenmp -DMOJO_OPM -DMOJO_AVX -msse4 -mavx2 -mrdrnd -I/opt/intel/oneapi/2024.1/opt/ippcp/tools/intel64

all: test train_mnist train_cifar

test: test.cpp 
	$(CC) $(CFLAGS_OMP) test.cpp -o test

train_mnist: train_mnist.cpp
	$(CC) $(CFLAGS_OMP) train_mnist.cpp -o train_mnist

train_cifar: train_cifar.cpp 
	$(CC) $(CFLAGS_OMP) train_cifar.cpp -o train_cifar

clean:
	-rm -f test
	-rm -f train_mnist
	-rm -f train_cifar
