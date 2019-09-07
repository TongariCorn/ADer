CC = g++
CFLAGS = -std=c++17 -Wall
SRCS = ader/*.cpp
RM = rm -rf

%: examples/%.cpp $(SRCS)
	$(CC) $(CFLAGS) $^

.PHONY: clean
clean:
	$(RM) a.out
