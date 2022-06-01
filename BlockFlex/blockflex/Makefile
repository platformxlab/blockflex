#############################################################################
#
# Makefile 
#
#===========================================================================

# Excecutable file.
PROGRAM   = harvest

# Parameters .
CC      = gcc 
#CFLAGS  = -g -O2 -Wall -pthread
CFLAGS  = -g -Wall -pthread
RM      = rm -f
COMPILE = $(CC) $(CFLAGS) -c
OBJS    = $(addsuffix .o, $(basename $(wildcard *.c)))
LINK_LIBS = -lrt

# Make goes here.
all: $(PROGRAM)
	@$(RM) $(OBJS)

debug: CFLAGS+=-DDEBUG
debug: all

clean:
	$(RM) $(OBJS) $(PROGRAM)
	$(RM) *.txt

show:
	@echo 'PROGRAM     :' $(PROGRAM)
	@echo 'SRCS        :' $(wildcard *.c)
	@echo 'OBJS        :' $(OBJS)
	@echo 'LINK.c      :' $(LINK.c)
	
# Object files.
objs:$(OBJS)
$(PROGRAM):$(OBJS)
	$(LINK.c)  $(OBJS) $(LINK_LIBS)  -o $@
%.o:%.c
	@$(COMPILE) $< -o $@

