g++ query_ball_point.cpp -o query_ball_point
/usr/local/cuda-10.1/bin/nvcc query_ball_point.cu -o query_ball_point_cuda
/usr/local/cuda-10.1/bin/nvcc query_ball_point_block.cu -o query_ball_point_block
/usr/local/cuda-10.1/bin/nvcc query_ball_point_grid.cu -o query_ball_point_grid
#/usr/local/cuda-10.1/bin/nvcc query_ball_point_grid_count.cu -o query_ball_point_grid_count
g++ -Wall selection_sort.cpp -o selection_sort
/usr/local/cuda-10.1/bin/nvcc selection_sort.cu -o selection_sort_cuda
