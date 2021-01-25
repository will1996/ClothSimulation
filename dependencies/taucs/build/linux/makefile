OSTYPE = linux
CONFIGURATION = test_timer
INCS = \
  build/linux/taucs_config_build.h\
  build/linux/taucs_config_tests.h\
  $(DIRSRC)taucs.h\
  $(DIRSRC)taucs_private.h\
  $(DIRBLD)taucs_config_tests.h
CILKC=$(CC)
CILKFLAGS=$(CFLAGS)
CILKOUTFLG=$(COUTFLG)


include config/$(OSTYPE).mk

LIBS = $(LIBLAPACK) $(LIBBLAS) $(LIBF77) $(LIBC)
default: all

include config/std.mk

build/linux/taucs_config_tests.h: $(DIROBJ)exists.log $(DIRPROGS)taucs_blas_underscore_test.c $(DIRPROGS)taucs_blas_nounderscore_test.c $(DIRPROGS)taucs_c99_complex_test.c $(DIRPROGS)taucs_cilk_test.c
	- $(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	  $(DIRPROGS)taucs_blas_underscore_test.c \
	  $(COUTFLG)$(DIROBJ)taucs_blas_underscore_test$(OBJEXT)
	- $(LD) $(LDFLAGS) \
	  $(LOUTFLG)$(DIROBJ)taucs_blas_underscore_test$(EXEEXT) \
	  $(DIROBJ)taucs_blas_underscore_test$(OBJEXT) $(LIBS)
	- $(DIROBJ)taucs_blas_underscore_test$(EXEEXT) build/linux/taucs_config_tests.h
	- $(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	  $(DIRPROGS)taucs_blas_nounderscore_test.c \
	  $(COUTFLG)$(DIROBJ)taucs_blas_nounderscore_test$(OBJEXT)
	- $(LD) $(LDFLAGS) \
	  $(LOUTFLG)$(DIROBJ)taucs_blas_nounderscore_test$(EXEEXT) \
	  $(DIROBJ)taucs_blas_nounderscore_test$(OBJEXT) $(LIBS)
	- $(DIROBJ)taucs_blas_nounderscore_test$(EXEEXT) build/linux/taucs_config_tests.h
	- $(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	  $(DIRPROGS)taucs_c99_complex_test.c \
	  $(COUTFLG)$(DIROBJ)taucs_c99_complex_test$(OBJEXT)
	- $(LD) $(LDFLAGS) \
	  $(LOUTFLG)$(DIROBJ)taucs_c99_complex_test$(EXEEXT) \
	  $(DIROBJ)taucs_c99_complex_test$(OBJEXT) $(LIBS)
	- $(DIROBJ)taucs_c99_complex_test$(EXEEXT) build/linux/taucs_config_tests.h
	- $(CILKC) -c $(CILKFLAGS) $(STDDEFS) $(STDINCS) \
	  $(DIRPROGS)taucs_cilk_test.c \
	  $(CILKOUTFLG)$(DIROBJ)taucs_cilk_test$(OBJEXT)
	- $(LD) $(LDFLAGS) \
	  $(LOUTFLG)$(DIROBJ)taucs_cilk_test$(EXEEXT) \
	  $(DIROBJ)taucs_cilk_test$(OBJEXT) $(LIBS)
	- $(DIROBJ)taucs_cilk_test$(EXEEXT) build/linux/taucs_config_tests.h
$(DIROBJ)taucs_logging$(OBJEXT): $(DIRSRC)taucs_logging.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_logging.c \
	$(COUTFLG)$(DIROBJ)taucs_logging$(OBJEXT)
$(DIROBJ)taucs_timer$(OBJEXT): $(DIRSRC)taucs_timer.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_timer.c \
	$(COUTFLG)$(DIROBJ)taucs_timer$(OBJEXT)
$(DIROBJ)taucs_ccs_base$(OBJEXT): $(DIRSRC)taucs_ccs_base.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_ccs_base.c \
	$(COUTFLG)$(DIROBJ)taucs_ccs_base$(OBJEXT)
$(DIROBJ)taucs_vec_base$(OBJEXT): $(DIRSRC)taucs_vec_base.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_vec_base.c \
	$(COUTFLG)$(DIROBJ)taucs_vec_base$(OBJEXT)
$(DIROBJ)taucs_ccs_ops$(OBJEXT): $(DIRSRC)taucs_ccs_ops.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_ccs_ops.c \
	$(COUTFLG)$(DIROBJ)taucs_ccs_ops$(OBJEXT)
$(DIROBJ)taucs_complex$(OBJEXT): $(DIRSRC)taucs_complex.c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	-DTAUCS_CORE_GENERAL \
	$(DIRSRC)taucs_complex.c \
	$(COUTFLG)$(DIROBJ)taucs_complex$(OBJEXT)
$(DIROBJ)$(CONFIGURATION)$(OBJEXT): $(DIRPROGS)$(CONFIGURATION).c $(INCS) $(STDDEPS)
	$(CC) -c $(CFLAGS) $(STDDEFS) $(STDINCS) \
	 $(DIRPROGS)$(CONFIGURATION).c \
	$(COUTFLG)$(DIROBJ)$(CONFIGURATION)$(OBJEXT)
libtaucs_content = \
  $(DIROBJ)taucs_logging$(OBJEXT) \
  $(DIROBJ)taucs_timer$(OBJEXT) \
  $(DIROBJ)taucs_ccs_base$(OBJEXT) \
  $(DIROBJ)taucs_vec_base$(OBJEXT) \
  $(DIROBJ)taucs_ccs_ops$(OBJEXT) \
  $(DIROBJ)taucs_complex$(OBJEXT) 
$(DIRLIB)libtaucs$(LIBEXT): $(libtaucs_content) $(STDDEPS)
	- $(RM) $(DIRLIB)libtaucs$(LIBEXT)
	$(AR) $(AOUTFLG)$(DIRLIB)libtaucs$(LIBEXT) $(libtaucs_content)
	$(RANLIB) $(DIRLIB)libtaucs$(LIBEXT)
$(DIREXE)$(CONFIGURATION)$(EXEEXT): $(STDDEPS) $(DIROBJ)$(CONFIGURATION)$(OBJEXT) $(DIRLIB)libtaucs$(LIBEXT)
	$(LD) $(LDFLAGS) \
	$(LOUTFLG)$(DIREXE)$(CONFIGURATION)$(EXEEXT) \
	$(DIROBJ)$(CONFIGURATION)$(OBJEXT) \
	-L$(DIRLIB) -ltaucs \
	$(LIBS)
all: $(STD_PRE_TARGETS) $(DIRLIB)libtaucs$(LIBEXT) $(DIREXE)$(CONFIGURATION)$(EXEEXT)
