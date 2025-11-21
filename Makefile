
LIBS := $(shell pkg-config --cflags --libs opencv4) -L./fftw-3.3.10/fftw-3.3.10/.libs/ -l:libfftw3.a -lm

.PHONY: all build run-all-images run-all-videos all-vids all-imgs

all: build run-all-images run-all-videos

all-vids: build run-all-videos

all-imgs: build run-all-images

build: main.cpp
	g++ main.cpp -fopenmp -O3 -o build/main $(LIBS)

run-all-images:
	./build/main -i ./input_images/lena.png -o ./output_images/lena.png
	./build/main -i ./input_images/lena.ppm -o ./output_images/lena.ppm
	./build/main -i ./input_images/tree.png -o ./output_images/tree.png
	./build/main -i ./input_images/tree.ppm -o ./output_images/tree.ppm
	./build/main -i ./input_images/tree.ppm -o ./output_images/tree.ppm
	./build/main -i ./input_images/humananflower.png -o ./output_images/humananflower.png

c-ffi:
	g++ -o ./build/libcconv3d.so conv3d.cpp -shared $(LIBS) -fPIC -fopenmp -O3

run-all-videos:
	./build/main -i ./input_videos/sample.mp4 -o ./output_videos/sample.mp4

clean:
	rm -rf build/
	mkdir -p build/
	rm -rf ./output_images/
	mkdir -p ./output_images/
	rm -rf ./output_videos/
	mkdir -p ./output_videos/
