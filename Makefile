CC := mpicc
CXX := mpic++
NVCXX := nvcc

NVFLAGS := -Idependencies/cuda-samples-master/Common -std=c++11 --expt-relaxed-constexpr

# # uncomment to disable OpenGL functionality
NO_OPENGL := true
NVLIBS :=  -L$(CUDA_PATH)/lib64 -lcurand -lcublas -lcusparse -lcusolver -lcudart

NVFLAGS_DEBUG  := -g -G -m64 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=compute_35

NVFLAGS_RELEASE  := -m64 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=compute_35

CXXFLAGS := -Idependencies/include -I/usr/local/include -I/opt/common/cuda/cuda-10.1.243/samples/common/inc/  -Wno-deprecated-declarations -std=c++11
ifdef NO_OPENGL
	CXXFLAGS := $(CXXFLAGS) -DNO_OPENGL
endif

CXXFLAGS_DEBUG := -Wall -g -Wno-sign-compare
CXXFLAGS_RELEASE := -O3 -Wreturn-type -openmp  -g 
LDFLAGS := -L/usr/local/opt/openblas/lib -Ldependencies/lib -L/opt/local/lib -L/usr/local/lib -L/lib64 -L/usr/local/lib/gcc/6 -L/usr/local/gfortran/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem-mt -lboost_thread-mt -ljson -lboost_system -lgomp -lalglib
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
	gpu.o \
	handle.o \
	init.o\
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
	save.o \
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
	dosimulation.o \
	gpusimulation.o \
	SH-combine.o\
	SpatialHashHelper.o\
	cudaCG.o\
	CCSManager.o


.PHONY: all debug release tags clean

release:

all: debug release ctags

debug: bin/arcsimd ctags

release: bin/arcsim ctags

bin/arcsimd: $(addprefix build/debug/,$(OBJ)) 
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS) 

bin/arcsim: $(addprefix build/release/,$(OBJ)) 
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)

build/debug/%.o: src/%.cpp 
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/release/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(CXXFLAGS_RELEASE) $< -o $@

build/release/%.o: interface/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS)  $(CXXFLAGS_RELEASE) $< -o $@

build/release/gpusimulation.o: src/cuda/gpusimulation.cu
	$(NVCXX) -c $(NVFLAGS) $(NVLIBS) $(NVFLAGS_DEBUG)  src/cuda/gpusimulation.cu -o build/release/gpusimulation.o

build/release/SH-combine.o: src/cuda/spatial-hashing/SH-combine.cu
	$(NVCXX) -c $(NVFLAGS) $(NVLIBS) $(NVFLAGS_DEBUG)  src/cuda/spatial-hashing/SH-combine.cu -o build/release/SH-combine.o

build/release/SpatialHashHelper.o: src/cuda/spatial-hashing/SpatialHashHelper.cu
	$(NVCXX) -c $(NVFLAGS) $(NVLIBS) $(NVFLAGS_DEBUG)  src/cuda/spatial-hashing/SpatialHashHelper.cu -o build/release/SpatialHashHelper.o

build/release/cudaCG.o: src/cuda/cudaCG.cpp
	$(NVCXX) -c $(NVFLAGS) $(NVLIBS) $(NVFLAGS_DEBUG)  src/cuda/cudaCG.cpp -o build/release/cudaCG.o


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
