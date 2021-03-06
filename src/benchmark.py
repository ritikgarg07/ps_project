import time

# function to time the loading of a datset
def timeit(ds, steps=1000):
  start = time.time()
  # it = iter(ds)
  for i in range(steps):
    batch = ds.get_batch()
    if i%10 == 0:
      # print(batch[0].shape)
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(10*steps/duration))
