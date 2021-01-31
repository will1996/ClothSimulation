
#CUDA_PATH ?= /usr/local/cuda-10.0
CUDA_PATH ?= /usr/local/cuda

HOST_COMPILER ?=g++

#NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCC          := nvcc #-ccbin $(HOST_COMPILER) -Xcompiler -fPIC -Xcompiler -fvisibility=hidden
NVFLAGS  := -I. -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc -Icuda-samples/Common -std=c++11 --expt-relaxed-constexpr

NVLIBS :=  -L$(CUDA_PATH)/lib64 -lcurand -lcublas -lcusparse -lcusolver -lcudart

NVFLAGS_RELEASE  := -m64 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=compute_35
NVFLAGS_DEBUG  := -g -G -m64 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_61,code=compute_61 -gencode arch=compute_35,code=compute_35

CC := gcc
CXX := mpic++
AR := ar

# # uncomment to disable OpenGL functionality
 NO_OPENGL := true
CXXFLAGS = 
#CXXFLAGS := -Idependencies/include -I. -I./inc -I./eigen -I/usr/include/jsoncpp
CXXFLAGS := -I. -Ieigen -Idependencies/include -Icuda-samples/Common -std=c++11


#CXXFLAGS := $(CXXFLAGS) #-DUSE_TIMER -DOUTPUT_TXT -DNO_UI 
#CXXFLAGS := $(CXXFLAGS) -DNO_UI -DFOR_DLL -DDLL_EXPORT -fvisibility=hidden 
#NVFLAGS := $(NVFLAGS) #-DUSE_TIMER -DOUTPUT_TXT -DNO_UI 
#NVFLAGS := $(NVFLAGS) -DNO_UI -DFOR_DLL -DDLL_EXPORT 

#CXXFLAGS_DEBUG := -D_DEBUG  -g 
#CXXFLAGS_RELEASE := -O3 
LDFLAGS :=   -Ldependencies/lib -lalglib -lgomp -ljson -lm -lglut -lGL

#LDFLAGS := -Ldependencies/lib -L/opt/local/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem -lboost_system -lboost_thread -ljson -lgomp -lalglib
#LDFLAGS := -Ldependencies/lib -L/opt/local/lib -lpng -lz -ltaucs -llapack -lblas -lboost_filesystem-mt -lboost_system-mt -lboost_thread-mt -ljson -lgomp -lalglib


OBJ := \
	auglag.o \
	conf.o \
	eigen.o \
	cloth.o \
	collision.o \
	constraint.o \
	dde.o \
	geometry.o \
	gpu.o\
	handle.o \
	init.o\
	io.o \
	magic.o \
	mesh.o \
	mot_parser.o \
	obstacle.o \
	save.o\
	spline.o \
	simulation.o \
	timer.o \
	tmbvh.o\
	transformation.o \
	util.o \
	vectors.o\
	\
	SH-combine.o\
	SpatialHashHelper.o\
	\
	cudaCG.o\
	cu-simulation.o\
	\
	CCSManager.o\
	\
	main.o

	#api-obj-viewer.o\
	#api-running.o
	#initModel.o\


	#DebugHash.o\
	SpatialHashCD.o\
	cudaAL.o\
	bbox.o\
	debug.o\
	lbvh.o\
	\
	cu-collision.o\
	proximity.o \
	physics.o \
#	collisionutil.o \
# bvh.o \
#	lsnewton.o \
#	dynamicremesh.o\
#	nearobs.o\
#	remesh.o\
#	tensormax.o\
#	popfilter.o \
#	separateobs.o \
#	equilibration.o\
#	obj-viewer.o\
	#taucs.o 

.PHONY: all debug release tags clean

release:

all: debug release ctags

#debug: bin/libgclothd.a ctags
debug: bin/gclothd ctags

#release: bin/libgcloth.so ctags
release: bin/gcloth ctags

bin/gcloth : $(addprefix build/release/,$(OBJ))
	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)

bin/libgcloth.so: $(addprefix build/release/,$(OBJ))
	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -Xcompiler -fPIC -shared -o $@

bin/gclothd: $(addprefix build/debug/,$(OBJ))
	$(NVCC) $(NVFLAGS) $(NVFLAGS_DEBUG) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)

bin/libgclothd.a: $(addprefix build/debug/,$(OBJ))
#	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)
#	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -shared -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)
	$(AR) -r $@ $^

bin/libgcloth.a: $(addprefix build/release/,$(OBJ))
#	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)
#	$(NVCC) $(NVFLAGS) $(NVFLAGS_RELEASE) $^ -shared -o $@ $(LDFLAGS) $(LDLIBS) $(NVLIBS)
	$(AR) -r $@ $^

build/debug/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/debug/%.o: interface/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/debug/%.o: src/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_DEBUG) $< -o $@

build/debug/%.o: cuda/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS) $(NVFLAGS) $(CXXFLAGS_DEBUG) $< -o $@

build/debug/%.o: cuda/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_DEBUG) $< -o $@

build/debug/%.o: spatial-hashing/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_DEBUG) $< -o $@

build/debug/%.o: adf/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_DEBUG) $< -o $@

build/release/%.o: src/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS)  $(CXXFLAGS_RELEASE) $< -o $@

build/release/%.o: interface/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS)  $(CXXFLAGS_RELEASE) $< -o $@

build/release/%.o: cuda/%.cpp
	$(CXX) -c $(CPPFLAGS) $(CXXFLAGS)  $(CXXFLAGS_RELEASE) $< -o $@

build/release/%.o: src/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_RELEASE) $< -o $@

build/release/%.o: cuda/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_RELEASE) $< -o $@

build/release/%.o: spatial-hashing/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_RELEASE) $< -o $@

build/release/%.o: adf/%.cu
	$(NVCC) -c $(NVFLAGS) $(NVFLAGS_RELEASE) $< -o $@

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
