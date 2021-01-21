CC := mpicc
CXX := mpic++
CUCXX = nvcc

CUFLAGS = -Idependencies/cuda-samples-master/Common
CUFLAGS_DEBUG :=  -g -arch sm_60 -rdc=true

# # uncomment to disable OpenGL functionality
NO_OPENGL := true

CXXFLAGS := -Idependencies/include -I/usr/local/include -Wno-deprecated-declarations
ifdef NO_OPENGL
	CXXFLAGS := $(CXXFLAGS) -DNO_OPENGL
endif
CXXFLAGS_DEBUG := -Wall -g -Wno-sign-compare
CXXFLAGS_RELEASE := -O3 -Wreturn-type -openmp -std=c++0x
LDFLAGS := -L/usr/local/opt/openblas/lib -Ldependencies/lib -L/opt/local/lib -L/usr/local/lib -L/usr/lib -L/usr/local/lib/gcc/6 -L/usr/local/gfortran/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem-mt -lboost_thread-mt -ljson -lboost_system -lgomp -lalglib
	LDFLAGS := $(LDFLAGS) -lglut 


OBJ := \
	auglag.o \
	bah.o \
	bvh.o \
	cloth.o \
	collision.o \
	collisionutil.o \
	conf.o \
	constraint.o \
	dde.o \
	dynamicremesh.o \
	geometry.o \
	handle.o \
	io.o \
	lbfgs.o \
	lsnewton.o \
	magic.o \
	main.o \
	mesh.o \
	misc.o \
	morph.o \
	mot_parser.o \
	nearobs.o \
	nlcg.o \
	obstacle.o \
	physics.o \
	popfilter.o \
	plasticity.o \
	proximity.o \
	remesh.o \
	refinemesh.o\
	runphysics.o \
	separate.o \
	separateobs.o \
	simulation.o \
	spline.o \
	strainlimiting.o \
	taucs.o \
	tensormax.o \
	timer.o \
	transformation.o \
	trustregion.o \
	util.o \
	vectors.o \
	dosimulation.o

CUOJB :=	simulation.o

.PHONY: all debug release tags clean

release:

all: debug release ctags

debug: bin/arcsimd ctags

release: bin/arcsim ctags

bin/arcsimd: $(addprefix build/debug/,$(OBJ)) 
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS) -lcudart

bin/arcsim: $(addprefix build/release/,$(OBJ)) cuda_dlinked.o 
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)  -lcudart -lcudadevrt

cuda_dlinked.o:  build/release/cuda/simulation.o 
	$(CUCXX) build/release/cuda/simulation.o  -o $@ -dlink  -lcudadevrt -lcudart -arch sm_60

build/debug/%.o: src/%.cpp 
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/debug/cuda/%.o: src/cuda/%.cu
	$(CUCXX) -c $(CUFLAGS) $(CUFLAGS_DEBUG) $< -o $@


build/release/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_RELEASE) $< -o $@

build/release/cuda/simulation.o: src/cuda/simulation.cu
	$(CUCXX) -c $(CUFLAGS) $(CUFLAGS_DEBUG) $< -o $@




# Nicked from http://www.gnu.org/software/make/manual/make.html#Automatic-Prerequisites
build/dep/%.d: src/%.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< -o $@.tmp; \
	sed 's,\($*\)\.o[ :]*,build/debug/\1.o build/release/\1.o: ,g' < $@.tmp > $@; \
	sed -e 's/#.*//' -e 's/^[^:]*: *//' -e 's/ *\\$$//' \
	    -e '/^$$/ d' -e 's/$$/ :/' < $@.tmp >> $@; \
	rm -f $@.tmp

-include $(addprefix build/dep/,$(OBJ:.o=.d))

ctags:
	cd src; ctags -w *.?pp
	cd src; etags *.?pp

clean:
	rm -rf bin/* build/debug/* build/release/*
