# build cluster expansion associated codes.
CC = gcc

#CFLAGS = -g -w
CFLAGS = -w

LFLAGS =
LIBS = -lm

INCLUDES = 

SRCS = clusterlist.c mat2d_prod.c vec_subs.c fnorm.c find_min.c sort_array.c find_recipro.c
OBJS = $(SRCS:.c=.o)

TARGET = run_clusterlist

default: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(TARGET) *.o
