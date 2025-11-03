
LIBS := $(shell pkg-config --cflags --libs opencv4) -L./fftw-3.3.10/fftw-3.3.10/.libs/ -l:libfftw3.a -lm

.PHONY: all build run-all-images run-all-videos all-vids all-imgs

all: build run-all-images run-all-videos

all-vids: build run-all-videos

all-imgs: build run-all-images

build: main.cpp
# 	g++ main.cpp -fopenmp -O3 -o ompmain.exe $(LIBS)
	g++ main.cpp -O3 -o main.exe $(LIBS)

run-all-images:
	rm -rf ./output_images/
	mkdir -p ./output_images/
	./main.exe -i ./input_images/lena.png -o ./output_images/lena.png
	./main.exe -i ./input_images/lena.ppm -o ./output_images/lena.ppm
	./main.exe -i ./input_images/tree.png -o ./output_images/tree.png
	./main.exe -i ./input_images/tree.ppm -o ./output_images/tree.ppm
	./main.exe -i ./input_images/tree.ppm -o ./output_images/tree.ppm
	./main.exe -i ./input_images/humananflower.png -o ./output_images/humananflower.png

run-all-videos:
	rm -rf ./output_videos/
	mkdir -p ./output_videos/
# 	./ompmain.exe -i ./input_videos/sample.mp4 -o ./output_videos/sample_omp.mp4
	./main.exe -i ./input_videos/sample.mp4 -o ./output_videos/sample.mp4
