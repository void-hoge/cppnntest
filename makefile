CC = g++
STD = -std=c++20
RELEASE = -O3 -mtune=native -march=native
PROG = nntest

$(PROG): main.o activation.o layer.o model.o
	$(CC) main.o activation.o layer.o model.o $(STD) $(RELEASE) -o $(PROG)

main.o: main.cpp
	$(CC) main.cpp -c $(STD) $(RELEASE) -o main.o

activation.o: activation.hpp activation.cpp
	$(CC) activation.cpp -c $(STD) $(RELEASE) -o activation.o

layer.o: layer.hpp layer.cpp
	$(CC) layer.cpp -c $(STD) $(RELEASE) -o layer.o

model.o: model.hpp model.cpp
	$(CC) model.cpp -c $(STD) $(RELEASE) -o model.o


clean:
	rm *.out $(PROG) *.o *~
