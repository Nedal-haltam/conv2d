.PHONY: build run-all

build: main.cpp
	g++ main.cpp -Wall -o main.exe

run-all:
	./main.exe ./input_images/lena.png ./output_images/edge_lena.png
	./main.exe ./input_images/lena.ppm ./output_images/edge_lena.ppm
	./main.exe ./input_images/tree.png ./output_images/edge_tree.png
	./main.exe ./input_images/tree.ppm ./output_images/edge_tree.ppm
	./main.exe ./input_images/tree.ppm ./output_images/edge_tree.ppm
	./main.exe ./input_images/humananflower.png ./output_images/edge_humananflower.png
