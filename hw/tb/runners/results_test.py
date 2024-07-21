import cocotb
from cocotb.triggers import RisingEdge
import csv

def read_csv_data(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(value) for value in row])
    return data

@cocotb.coroutine
async def read_ram(dut, start_address, rows, cols):
    ram_data = []
    for row in range(rows):
        row_data = []
        for col in range(cols):
            await RisingEdge(dut.clk)  # Synchronize with the clock
            address = start_address + row * cols + col
            row_data.append(float(dut.ram[address].value))
        ram_data.append(row_data)
    return ram_data

@cocotb.test()
async def test_ram_data(dut,expected_file,rows,cols,tolerance,start_address =0):

    expected_data = read_csv_data(expected_file)
    ram_data = await read_ram(dut, start_address, rows, cols)
    
    for i in range(rows):
        for j in range(cols):
            assert abs(ram_data[i][j] - expected_data[i][j]) <= tolerance, \
                f"Mismatch at row {i}, column {j}: RAM data {ram_data[i][j]}, expected {expected_data[i][j]}"
    
    cocotb.log.info("RAM data matches the expected output within the tolerance level.")
