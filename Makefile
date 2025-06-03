

ARGS= .\image.ppm .\output.ppm

all: build run

build: main.c png2ppm.c
	gcc png2ppm.c -Wall -Wextra -Wpedantic -o png2ppm.exe
	gcc main.c -Wall -Wextra -Wpedantic -o conv.exe

run: conv.exe
	.\conv.exe $(ARGS)

clean:
	del /Q conv.exe 2>nul || exit 0