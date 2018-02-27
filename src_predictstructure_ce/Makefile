# build cluster expansion associated codes.
CC = mpicc

#CFLAGS = -g -w
CFLAGS = -w

LFLAGS =
LIBS = -lm

INCLUDES =

SRCS = predictstructure_MC_ternary_mag_par.c mat2d_prod.c obtain_corr_mat_mag_par.c mat2d_sum_row.c sort_array.c sort_int_array.c
 
OBJS = $(SRCS:.c=.o)

TARGET = run_predictstructure_ce

default: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LFLAGS) $(LIBS)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

clean:
	$(RM) $(TARGET) *.o
