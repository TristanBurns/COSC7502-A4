NVCC?=nvcc
NVFLAGS?=-O0 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets

%.o : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) -o $@

clean:
	rm GeneticAlgorithm
	rm outputfile.txt