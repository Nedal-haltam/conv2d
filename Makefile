

build: main.c
	gcc main.c -Wall -Wextra -Wpedantic -o conv.exe

run:
	./conv.exe ./lena.png ./edge_lena.png
	./conv.exe ./lena.ppm ./edge_lena.ppm
	./conv.exe ./tree.png ./edge_tree.png
	./conv.exe ./tree.ppm ./edge_tree.ppm
