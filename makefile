FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 

p_nbody: nbody_parallel.o compute_parallel.o
	nvcc $(FLAGS) $^ -o $@ $(LIBS)
nbody_parallel.o: nbody_parallel.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 
compute_parallel.o: compute_parallel.cu config.h vector.h $(ALWAYS_REBUILD)
	nvcc $(FLAGS) -c $< 

clean:
	rm -f *.o nbody 
