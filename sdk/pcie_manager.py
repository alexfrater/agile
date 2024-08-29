import os
import pypci

class PCIe_Manager:
  def __init__(self):
    # self.pcie_device = self.connect_to_device()
    self.pcie_device = "/dev/xdma0_user"

  def connect_to_device(self):
    devices  = pypci.lspci()
    for device in devices:
        # print('device',device)
        if device.vendor_id ==  0x10EE:
            print('Xilinx device found:')
            xilinx_device = device
            break
    return xilinx_device

  def pci_rescan(self):
      try:
          # Write to the rescan file in the sysfs
          with open('/sys/bus/pci/rescan', 'w') as f:
              f.write('1')
          print("PCI bus rescan triggered successfully.")
      except PermissionError:
          print("Permission denied: You need to run this script as root to rescan the PCI bus.")
      except Exception as e:
          print(f"An error occurred while rescanning the PCI bus: {e}")


  def read_from_memory(self, address):
    size = 32 #TODO set in init
    """Read data from the specified address in memory."""
    with open(device, 'rb') as f:
        f.seek(address)
        data = f.read(size)
        print(f"Data read from address {hex(address)}: {data.hex()}")
        return data


  def write_to_memory(self,address, data):
      with open(self.pcie_device, 'r+b') as f:
          f.seek(address)
          f.write(data)
          print(f"Data written to address {hex(address)}")

  def read_memory_file(self,filepath):
      with open(filepath, 'r') as file:
          hex_string = file.read().strip().replace("\n", "")
      
      # Convert hex string to bytes
      data_bytes = bytes.fromhex(hex_string)
      return data_bytes

  def write_file(self,filepath,address):
    # Load the memory data from the file
    memory_data = self.read_memory_file(filepath)
    # Write the entire memory data to the specified address
    self.write_to_memory(address, memory_data)


