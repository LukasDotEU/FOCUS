import time

class Timer:
    """
    A simple context manager for timing code blocks.
    Usage:
      with Timer() as t:
          do_something()
      elapsed = t.elapsed  # in seconds
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
