from aqx.backends.multi import BACKENDS, get_backend, prepare_inputs

workload = "monte_carlo"
size = 1024
args, kwargs = prepare_inputs(workload, size)

for backend in BACKENDS:
    value = get_backend(workload, backend)(*args, **kwargs)
    print(f"{backend:>16}: {float(value):.8f}")
