# build cluster expansion associated codes.
CC = mpicc

#CFLAGS = -g -w
CFLAGS = -w

LFLAGS = -L/opt/local/lib
LIBS = -lm -lgsl -lgslcblas

INCLUDES = -I/opt/local/include

SRCS = findcluster_ternary_par.c obtainerr2_par.c least_square_solver.c mat2d_prod.c 
OBJS = $(SRCS:.c=.o)

TARGET = run_findcluster

default: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(TARGET) *.o
