"""Contains decorators."""
import os
import concurrent.futures as cf


def threads(*margs, random_kwarg=None, **mkwargs):
    """Decorator for multithreading.

    Parameters
    ----------
    random_kwarg : dict or None
        Contains name of random variable to be passed to decorated function
        and method for its generation. Default to None.
    """
    def threads_decorator(method):
        def wrapper(self, *args, **kwargs):
            results = []

            if random_kwarg is not None:
                random_values = random_kwarg['generator'](size=len(self.indices))

            if 'n_workers' not in mkwargs:
                n_workers = os.cpu_count() * 4

            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for i in range(len(self.indices)):
                    if random_kwarg is not None:
                        kwargs[random_kwarg['kwarg']] = random_values[i]
                    res = executor.submit(method, self, i, *args, **kwargs)
                    futures.append(res)
                cf.wait(futures, timeout=None, return_when=cf.ALL_COMPLETED)
            results = [f.result() for f in futures]
            if any(isinstance(res, Exception) for res in results):
                errors = [error for error in results if isinstance(error, Exception)]
                print(errors)
                return self

            return self
        return wrapper
    return threads_decorator
