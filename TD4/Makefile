demo:
	export PATH=/usr/local/cuda/bin:$PATH
	nvcc add.cu -o demo -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

sum:
	export PATH=/usr/local/cuda/bin:$PATH
	nvcc sum.cu -o sum -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

variance:
	export PATH=/usr/local/cuda/bin:$PATH
	nvcc variance.cu -o variance -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

sum_distances:
	export PATH=/usr/local/cuda/bin:$PATH
	nvcc sum_distances.cu -o sum_distances -arch=sm_60 -std=c++11 -I/usr/local/cuda/include

clean:
	rm -f demo
	rm -f sum_distances
	rm -f sum
	rm -f variance
