from ipykernel.kernelapp import IPKernelApp
from . import REPLKernel
IPKernelApp.launch_instance(kernel_class=REPLKernel)
