import torch 
# x = 0
# print(torch.cuda.is_available(),
# torch.cuda.device_count(),
# torch.cuda.current_device(),
# torch.cuda.device(x),
# torch.cuda.get_device_name(x)
# )

# t = torch.cuda.get_device_properties(x).total_memory
# r = torch.cuda.memory_reserved(x)
# a = torch.cuda.memory_allocated(x)
# f = r-a  # free inside reserved
# print(f)
print(torch.cuda.mem_get_info())
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    total_memory = torch.cuda.get_device_properties(i).total_memory
    allocated_memory = torch.cuda.memory_allocated(i)
    free_memory = total_memory - allocated_memory
    name = torch.cuda.get_device_name(i)
    print(f'GPU {i}:')
    print(f'Ten thiet bi: {name}')
    print(f'Tổng bộ nhớ: {total_memory / 1024**3:.2f} GB')
    print(f'Bộ nhớ đã dùng: {allocated_memory / 1024**3:.2f} GB')
    print(f'Bộ nhớ còn lại: {free_memory / 1024**3:.2f} GB\n')