import torch
import subprocess, multiprocessing, time
import numpy as np
import os
import subprocess
import shutil
import pandas as pd

def read_power_file(filename):
    powers = []
    with open(filename, "r") as file:
        for line in file:
            power = float(line.strip())
            powers.append(power)
    return powers

def read_timing_file(filename):
    lst = []
    with open(filename, "r") as file:
        for line in file:
            lst = [float(a) for a in line.split(",")]
    return lst

class BenchmarkWrapper():
    def __init__(self, model):
        self.model = model
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, x, edge_index, edge_attr):
        model_input = (x, edge_index)
        if edge_attr is not None:
            model_input = model_input + (edge_attr,)  
        out = self.model(*(model_input))
        return out

    def predict(self, batch):
        x, edge_index,edge_attr = batch[0], batch[1], batch[2]
        
        torch.cuda.empty_cache()
        torch.cuda._sleep(1_000_000)
        self.starter.record()
        
        _ = self.forward(x, edge_index,edge_attr)
        torch.cuda.synchronize()
        self.ender.record()
        torch.cuda.synchronize()
        inference_time = self.starter.elapsed_time(self.ender) /1000.0
        return inference_time 

    def warm_up(self, batch, steps=10):
        x, edge_index = batch[0], batch[1]
        for _ in range(steps):
            out = self.forward(x, edge_index)
        return out
    


class CPUBenchmarkWrapper():
    def __init__(self, model):
        self.model = model
        # self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.loss_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    # def forward(self, x, edge_index,edge_attr):
    #     out = self.model(x, edge_index,edge_attr)
    #     return out


    def forward(self, x, edge_index, edge_attr):
        model_input = (x, edge_index)
        if edge_attr is not None:
            model_input = model_input + (edge_attr,)  
        out = self.model(*(model_input))
        return out


    def predict(self, batch):
        x, edge_index,edge_attr = batch[0], batch[1], batch[2]
        start_time = time.time()
        with torch.no_grad():  # Disable gradient calculation
            _ = self.forward(x, edge_index,edge_attr)
        end_time = time.time()
        inference_time = end_time - start_time
        return inference_time
    



class BenchmarkingManager:
    def __init__(self, model, args = None, inputs= None, graph = None):
        if (torch.cuda.is_available()):
            self.bman = BenchmarkWrapper(model)
        else:
            self.bman = CPUBenchmarkWrapper(model) #Temp
        self.args = args
        
        self.graph = graph
        #TODO just use args
        self.cpu = False if args.cpu is None else args.cpu
        self.gpu = False if args.gpu is None else args.gpu
        self.sim = False if args.sim is None else args.sim
        self.fpga_clk_freq = 200e6 if args.fpga_clk_freq is None else args.fpga_clk_freq
        self.args = None if args is None else args
        self.device = None if args.device is None else args
        self.preload = False if args.preload is None else args.preload
        self.gui =True
        # self.metrics = False if args.metrics is None else args.metrics
        self.model = model


    def gpu_run_inference(self):
        print(f"device {self.device}")
        self.bman.model.to(torch.device(f"cuda:{self.device}"))
        data = self.graph.dataset
        data.x = data.x.to(torch.device(f"cuda:{self.device}"))
        data.edge_index = data.edge_index.to(torch.device(f"cuda:{self.device}"))
        data.edge_attr = data.edge_attr.to(torch.device(f"cuda:{self.device}"))

        times = []
        for i in range(1000):
            time_taken = self.bman.predict(batch=(data.x, data.edge_index, data.edge_attr))
            times.append(time_taken)

        avg_time = np.mean(times)
        std_dev = np.std(times)
        with open("timing_tmp.txt", "w") as f:
            f.write(f"{avg_time}, {std_dev}")
        return avg_time


    def gpu_measure_power(self):
        with open("powers.txt", "w") as file:
            pass

        while True:
            try:
                power_output = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
                power = float(power_output.decode().strip())
                with open("powers.txt", "a") as file:
                    file.write(f"{power}\n")
                time.sleep(0.1)
            except KeyboardInterrupt:
                print(f"finishing")
    
    def cpu_benchmark(self):
        self.bman.model.to(torch.device("cpu"))
        data = self.graph.dataset
        data.x = data.x.to(torch.device("cpu"))
        data.edge_index = data.edge_index.to(torch.device("cpu"))
        
        times = []
        for i in range(100):
            time_taken = self.bman.predict(batch=(data.x, data.edge_index,data.edge_attr))
            times.append(time_taken)

        avg_time = np.mean(times)
        std_dev = np.std(times)
        throughput = self.graph.dataset.y.shape[0] / avg_time

        return {
            "cpu_latency_mean": avg_time,
            "cpu_latency_std_dev": std_dev,
            "cpu_nodes_per_ms": throughput

        }

    def gpu_benchmark(self):
        inference_job = multiprocessing.Process(target=self.gpu_run_inference)
        power_job = multiprocessing.Process(target=self.gpu_measure_power)

        inference_job.start()
        power_job.start()

        try:
            inference_job.join()  # Wait for inference_job process to finish
            lst = read_timing_file("timing_tmp.txt")
            print(f"Inference job completed in {lst}ms. Terminating power job...")
            power_job.terminate()  # Terminate power_job process

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Terminating processes...")
            inference_job.terminate()
            power_job.terminate()

        power = np.mean(read_power_file("powers.txt"))
        print(f"mean power {power}")
        
        throughput = self.graph.dataset.y.shape[0] / lst[0]
        return {
                "gpu_latency_mean": lst[0],
                "gpu_latency_std_dev": lst[1],
                "gpu_mean_power": power,
                "gpu_nodes_per_ms": throughput,
                "gpu_throughput_per_watt": throughput/power
                }

    def fpga_benchmark(self):

        # * Load layer config
        if (self.args.preload):
            dest_dir = os.environ.get(f"WORKAREA") + "/hw/sim/layer_config"
            config_path = f"{self.args.preload_path}/layer_configs/layer_config_degree_{self.graph.avg_degree}_nodes_{self.graph.num_nodes}"
            
            assert os.path.isdir(config_path), f"{config_path} was not found"

            # Delete current layer config
            try:
                shutil.rmtree(dest_dir)
            except OSError:
                pass

            # Copy new layer config
            cm = f"cp -r {config_path} {dest_dir}"
            print(f"==== Running {cm}")
            subprocess.run(cm, shell=True, capture_output=False, text=True)

        os.environ['AMPLE_GRAPH_TB_TOLERANCE'] = str(self.args.tb_tolerance)
        os.environ['AMPLE_GRAPH_TB_LOG_LEVEL'] = str(self.args.tb_log_level)
        os.environ['AMPLE_GRAPH_TB_NODESLOT_COUNT'] = '64'
        os.environ['AMPLE_GRAPH_TB_MODEL_NAME'] = str(self.model.__class__.__name__)


        # * Run simulation (assume )
        path = os.environ.get("WORKAREA") + "/hw/sim"
        print(f"cd {path}")
        command = ""
        if (self.args.build):
            print(f"Building")
            command += f"cd {path}; make build"
            command += '&& '
        if (self.args.gui):
            command += f"cd {path}; make run_simgui"

        else:
            command += f"cd {path}; make run_sim"

        print(f"==== Running command: {command}")
        process = subprocess.run(command, shell=True, capture_output=False, text=True)

        with open(f"{path}/sim_time.txt", "r") as f:
            stime = float(f.readline())

        if self.args.metrics:
            cycles_dict = self.read_cycles_file(f"{path}/sim_cycles.txt")
            sim_cycle_time = sum(cycles_dict.values()) * (1/self.fpga_clk_freq)
            throughput = self.graph.dataset.y.shape[0] / float(sim_cycle_time)
            mean_power = 30.0

            metrics  = {
                "fpga_latency": stime,
                "fpga_sim_cycle_time": sim_cycle_time,
                "fpga_mean_power": mean_power,
                "fpga_nodes_per_ms": throughput,
                "fpga_throughput_per_watt": throughput/mean_power
            }
        else:
            metrics = {
                "fpga_latency": stime
            }



        # print(f"Metrics: {metrics}")
        return metrics

    def read_cycles_file(self,file_path):
        cycles_dict = {}

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("Layer"):
                parts = line.split()
                layer = int(parts[1])  # Extract layer number
                cycles = int(parts[3])  # Extract cycle count
                cycles_dict[layer] = cycles

        return cycles_dict

    def benchmark(self):
        metrics = {}
        if (self.cpu):
            print('1')

            metrics["cpu"] = self.cpu_benchmark()
        if (self.gpu):
           metrics["gpu"] = self.gpu_benchmark()
        if (self.sim):
            metrics["fpga"] = self.fpga_benchmark()
        return metrics

    def print_results(self, metrics):
        rows = []
        for component, values in metrics.items():
            for metric, value in values.items():
                formatted_metric = metric.replace("_", " ").replace("-", " ").title()
                formatted_value = f"{value:.6f}" if isinstance(value, float) else f"{value:.6f}"
                rows.append([component, formatted_metric, formatted_value])

        # Create a DataFrame and print it
        df = pd.DataFrame(rows, columns=["Component", "Metric", "Value"])
        print(df.to_markdown(index=False))
    
