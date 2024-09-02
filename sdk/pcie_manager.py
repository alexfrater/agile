import os
# import pypci

class PCIeManager:
  def __init__(self):
    # self.pcie_device = self.connect_to_device()
    
    #Regs
    self.regs_write = fw = os.open('/dev/xdma0_user',os.O_WRONLY)
    self.regs_read = fw = os.open('/dev/xdma0_user',os.O_RDONLY)

    #DDR memory
    self.mem_write = os.open('/dev/xdma0_h2c_0',os.O_WRONLY)
    self.mem_read = os.open('/dev/xdma0_c2h_0',os.O_RDONLY)

    xdma_files = {
        'regs_write': self.regs_write,
        'regs_read': self.regs_read,
        'mem_write': self.mem_write,
        'mem_read': self.mem_read,

    }


    def __del__(self):
        for name, fd in xdma_files.items():
            if fd is not None: 
                try:
                    os.close(fd)
                    print(f"File descriptor '{name}' closed successfully.")
                except OSError as e:
                    print(f"Error closing file descriptor '{name}': {e}")
                finally:
                    file_descriptors[name] = None  


    
  def connect_to_device(self):
    devices  = pypci.lspci()
    for device in devices:

        if device.vendor_id ==  0x10EE:
            print('device',device)
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


  
  def write_to_memory(self,address, datab):
    os.pwrite(self.mem_write,datab,address)
  
  def read_from_memory(self,address, num_bytes):
    return os.pread(self.mem_read,num_bytes,address)


    #Add offsets
  def write_to_reg(self,address, datab):
    os.pwrite(self.regs_write,datab,address)

  
  def read_from_reg(self,address): #32bit reg
    return os.pread(self.regs_read,4,address)

  def read_host_memory_file(self,filepath):
      with open(filepath, 'r') as file:
          hex_string = file.read().strip().replace("\n", "")
      
      data_bytes = bytes.fromhex(hex_string)
      return data_bytes

  def write_file(self,filepath,address):
    memory_data = self.read_host_memory_file(filepath)
    self.write_to_memory(address, memory_data)
    print('Written to memory', len(memory_data), 'bytes from file:', filepath)


