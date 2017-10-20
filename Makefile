CC = g++
CFLAGS = -Wall -O3 -fopenmp
LDFLAGS = -lgomp

all: beer_seg

beer_seg.o: beer_seg.cpp
	$(CC) $(CFLAGS) -c beer_seg.cpp munkres/src/munkres.cpp

beer_seg: beer_seg.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o beer_seg beer_seg.o munkres/src/munkres.cpp

clean:
	rm -f beer_seg beer_seg.o munkres.o
