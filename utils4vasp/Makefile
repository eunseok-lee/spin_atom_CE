# build cluster expansion associated codes.
CC = gcc

#CFLAGS = -g -w
CFLAGS = -w

LFLAGS =
LIBS = -lm

INCLUDES = 

SRCS1 = extend2largercell.c 
SRCS2 = vasp2datamag.c
 
OBJS1 = $(SRCS1:.c=.o)
OBJS2 = $(SRCS2:.c=.o)

TARGET1 = extend2largercell
TARGET2 = vasp2datamag 

default: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJS1)
	$(CC) $(CFLAGS) -o $(TARGET1) $(OBJS1) $(LFLAGS) $(LIBS)

$(TARGET2): $(OBJS2)
	$(CC) $(CFLAGS) -o $(TARGET2) $(OBJS2) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(TARGET1) $(TARGET2) *.o
