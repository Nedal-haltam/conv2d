

ARGS= .\image.ppm .\output.ppm

all: build run

build: main.c
	gcc main.c -Wall -Wextra -Wpedantic -o conv.exe

run: conv.exe
	.\conv.exe $(ARGS)

clean:
	del /Q conv.exe 2>nul || exit 0