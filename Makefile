.PHONY: all build run-all-images run-all-videos all-vids all-imgs

all: build run-all-images run-all-videos

all-vids: build run-all-videos

all-imgs: build run-all-images

build: main.cpp
	g++ main.cpp -fopenmp -O3 -o ompmain.exe $(shell pkg-config --cflags --libs opencv4)
	g++ main.cpp -O3 -o main.exe $(shell pkg-config --cflags --libs opencv4)

run-all-images:
	rm -rf ./output_images/
	mkdir -p ./output_images/
	./main.exe ./input_images/lena.png ./output_images/lena.png
	./main.exe ./input_images/lena.ppm ./output_images/lena.ppm
	./main.exe ./input_images/tree.png ./output_images/tree.png
	./main.exe ./input_images/tree.ppm ./output_images/tree.ppm
	./main.exe ./input_images/tree.ppm ./output_images/tree.ppm
	./main.exe ./input_images/humananflower.png ./output_images/humananflower.png

run-all-videos:
	rm -rf ./output_videos/
	mkdir -p ./output_videos/
	./ompmain.exe ./input_videos/sample.mp4 ./output_videos/sample_omp.mp4
	./main.exe ./input_videos/sample.mp4 ./output_videos/sample.mp4
