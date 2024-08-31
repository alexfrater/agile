import argparse
import pandas as pd
import torch.nn as nn


from sdk.ample_driver import Ample_Driver
from sdk.ample_compiler import AmpleCompiler
from sdk.benchmarking_manager import BenchmarkingManager

class Ample():
    def __init__(self,
        sim =False,
        name="ample",
        index=None,
        node_slots = 32,
        message_channel_count = 16,
        precision_count = 1,
        aggregation_buffer_slots = 4
    ):
        self.name = name
        self.model = None
        self.sim = sim
        self.add_to_device_method()
        self.compiler = AmpleCompiler(sim = sim)
        if not self.sim:
            self.ample = self.connect_to_device()

    def add_to_device_method(self):
        ample_instance = self 
        def extended_to_device(model, device, data=None): #Figure out if there is a way to change this
            ample_instance.to_device(model, device,data)
        nn.Module.to_device = extended_to_device

    def to_device(self, model, device,data=None):
        self.model = model
        self.inputs = data
        if device == 'ample':
            self.compiler.compile(model,data=data,trace_mode='hooks')
            self.overload_forward()
        else:
            print(f'Moving model to {device}')
            model.to(device) 

    def overload_forward(self):
        original_forward = self.model.forward

        def ample_forward(*args, **kwargs):
            print("Executing on AMPLE")

            if self.sim:
                self.simulate() 
            else:
                self.driver.load_layer_config()
                self.driver.load_regbanks()
                self.driver.execute()


            return original_forward(*args, **kwargs)

        self.model.forward = ample_forward
    
    def simulate(self,cpu = False, gpu = False):
        args = argparse.Namespace(
            cpu=cpu,
            gpu=gpu,
            sim = True,
            fpga_clk_freq = 250e6,
            device = 'ample',
            preload = False,
            tb_tolerance = 0.01,
            tb_log_level = 'DEBUG',
            build = False,
            gui = False,
            metrics = False,
        )

        bman = BenchmarkingManager(inputs=self.inputs, model=self.model, args=args)
        # if cpu:
        #     metrics = bman.benchmark_cpu()
        # if gpu:
        #     metrics = bman.benchmark_gpu()

        metrics = bman.benchmark()
        #TODO use bman results table
        rows = []
        for component, values in metrics.items():
            for metric, value in values.items():
                formatted_metric = metric.replace("_", " ").replace("-", " ").title()
                formatted_value = f"{value:.6f}" if isinstance(value, float) else f"{value:.6f}"
                rows.append([component, formatted_metric, formatted_value])

        # Create a DataFrame and print it
        df = pd.DataFrame(rows, columns=["Component", "Metric", "Value"])
        print(df.to_markdown(index=False))

   


  