
import numpy as np
import logging
import os
from .utilities import int_list_to_byte_list, float_list_to_byte_list

import torch
from torch_geometric.nn import GCNConv, GINConv, SAGEConv
from torch.nn import Linear

from .models.models import GraphSAGE_Model, Edge_Embedding_Model, AGG_MLP_Model, AggregateEdges, Interaction_Net_Model

class Memory_Mapper:

    def __init__(self, graph, model, base_path="config_files", dump_file="memory.mem"):
        self.graph = graph
        self.model = model
        self.sub_memory_hex = []
        self.num_layers = self.count_layers()
        weights_list = [0]*self.num_layers
        out_messages_list = [0]*self.num_layers #Can remove - using ptr
        #Used to change adj list between layers
        adj_list = {'nbrs': 0, 'edge_nbrs': 0, 'self_ptr': 0 } # Change name perhaps to layer offset adj
        self.offsets = {'adj_list': adj_list, 'scale_factors': 0, 'in_messages':0, 'weights':weights_list, 'out_messages': out_messages_list, 'edge_attr_messages':0}
        self.out_messages_ptr = 0
        
        self.dump_file = os.path.join(base_path, dump_file)

        if isinstance(self.model, Interaction_Net_Model) or isinstance(self.model, Edge_Embedding_Model): #TODO put in initializer and then pass
            self.edge_attr = 1
        else:
            self.edge_attr = 0

        #Temp
        self.weight_list = []
        self.weight_addr_i = 0
        self.out_feature_count = None
    def map_memory(self,memory_ptr=0,in_messages_addr = None,edge_attr_messages_addr = None):
        self.memory_ptr = memory_ptr
        logging.debug(f"Mapping memory contents.")
        self.sub_memory_hex = []
        self.map_in_messages(in_messages_addr,edge_attr_messages_addr)
        self.map_adj_list()
        self.map_scale_factors()
        self.map_weights()
        self.map_out_messages() #TODO change to map the actual out messages - currently returns start of intermediate data


    #TODO Change to use messages from previous sub_models or do init manager using layer offset
    def map_in_messages(self,in_messages_addr,edge_attr_messages_addr):
        if in_messages_addr is not None:
            self.offsets['in_messages'] = in_messages_addr 
           
        else:
            self.offsets['in_messages'] = len(self.sub_memory_hex) + self.memory_ptr

            for node in self.graph.nodes:
                self.sub_memory_hex += float_list_to_byte_list(self.graph.nodes[node]["meta"]['embedding'], align=True, alignment=64)

        #TODO:Change Location of edge attributes
        if edge_attr_messages_addr is not None:
            self.offsets['edge_attr_messages'] = edge_attr_messages 
        else:
            self.offsets['edge_attr_messages'] = len(self.sub_memory_hex) + self.memory_ptr

            if self.edge_attr:
                for edge in self.graph.edges:
                    edge_data = self.graph.edges[edge]["meta"] 
                    if 'embedding' in edge_data:
                        # print('edge embedding',edge_data['embedding'])
                        self.sub_memory_hex += float_list_to_byte_list(self.graph.edges[edge]["meta"]['embedding'], align=True, alignment=64)


    def map_adj_list(self):
        #TODO change adj list so it can be switched out and only serve a node group instead of all
        self.offsets['adj_list']['nbrs'] = len(self.sub_memory_hex) + self.memory_ptr
        for node in self.graph.nodes:
            node_metadata = self.graph.nodes[node]['meta']
            self.sub_memory_hex += int_list_to_byte_list(node_metadata['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right")
        

        if self.edge_attr:
            for (u,v) in self.graph.edges():
                edge_metadata = self.graph[u][v]['meta']
                print('edge_metadata',edge_metadata)
                self.sub_memory_hex += int_list_to_byte_list(edge_metadata['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right")




        # self.offsets['adj_list']['edge_nbrs'] = len(self.sub_memory_hex)
        # for node in self.graph.nodes:
        #     node_metadata = self.graph.nodes[node]['meta']
        #     self.sub_memory_hex += int_list_to_byte_list(node_metadata['edge_message_ptrs'], align=True, alignment=64, pad_side="right")
        
        


        #if linear layers
        self.offsets['adj_list']['self_ptr'] = len(self.sub_memory_hex) + self.memory_ptr
        for node in self.graph.nodes:
            node_metadata = self.graph.nodes[node]['meta']
            self.sub_memory_hex += int_list_to_byte_list(node_metadata['self_ptr'], align=True, alignment=64, pad_side="right")

        if self.edge_attr:
            for (u,v) in self.graph.edges():
                edge_metadata = self.graph[u][v]['meta']
                self.sub_memory_hex += int_list_to_byte_list(edge_metadata['self_ptr'], align=True, alignment=64, pad_side="right")

        #Node update


        self.offsets['adj_list']['concat_ptr'] = len(self.sub_memory_hex) + self.memory_ptr
        last_node_key = max(self.graph.nodes.keys())

        concat_offset = self.graph.nodes[last_node_key]['meta']['self_ptr'][0]+128 #TODO change this

        for node in self.graph.nodes:
            #TODO for x in len concat width etc
            node_metadata = self.graph.nodes[node]['meta']

            concat_ptr = [node_metadata['self_ptr'][0], (node_metadata['self_ptr'][0]+concat_offset)]

            self.sub_memory_hex += int_list_to_byte_list(concat_ptr, align=True, alignment=64, pad_side="right")

      
   
    # def calc_axi_addr(self,feature_count):
    #     return math.ceil(4*feature_count / data_width) *data_width

    def map_scale_factors(self):
        self.offsets['scale_factors'] = len(self.sub_memory_hex) + self.memory_ptr

        for node in self.graph.nodes:
            self.graph.nodes[node]["meta"]['scale_factors_address'] = len(self.sub_memory_hex) + self.memory_ptr
            if (self.graph.nodes[node]["meta"]['precision'] == 'FLOAT_32'):
                self.sub_memory_hex += float_list_to_byte_list(self.graph.nodes[node]["meta"]['scale_factors'], align=True, alignment=64, pad_side='left')
            else:
                self.sub_memory_hex += int_list_to_byte_list(self.graph.nodes[node]["meta"]['scale_factors'], align=True, alignment=64, pad_side='left')
        
        # Set offset for next memory range

    def map_weights(self):
        self.offsets['weights'][0] = len(self.sub_memory_hex)  + self.memory_ptr
        for idx,layer in enumerate(self.model.layers):
            # print('-----layer---j',idx,layer.name)
            if isinstance(layer, GCNConv):
                linear = layer.lin
            elif isinstance(layer, GINConv):
                linear = layer.nn
            elif isinstance(layer, SAGEConv):
                linear = layer.lin_l
            elif isinstance(layer, Linear):
                print('linear')
                linear = layer
            elif isinstance(layer, AGG_MLP_Model):
                linear = layer.lin
            elif isinstance(layer, AggregateEdges):
                linear = layer.lin
            else:
                raise RuntimeError(f"Unrecognized layer {layer}")

            self.out_feature_count = linear.weight.shape[0]
            for outf in range(self.out_feature_count):
                print('memory mapper weight',outf)
                print('linear weight',linear.weight[outf])
                self.sub_memory_hex += float_list_to_byte_list(linear.weight[outf], align=True, alignment=64)
                self.weight_list.append(float_list_to_byte_list(linear.weight[outf], align=True, alignment=64))
                if outf == 27:
                    self.weight_addr_i = len(self.sub_memory_hex)//64

                # if outf == 27:
                #     self.weight_addr_0 = len(self.sub_memory_hex)//64
            if(idx < self.num_layers-1):
                self.offsets['weights'][idx+1] = len(self.sub_memory_hex)  + self.memory_ptr
        # Set offset for next memory range


    def map_out_messages(self):
        self.offsets['out_messages'][0] = len(self.sub_memory_hex)  + self.memory_ptr
        #ADD space for messages to go
        #TODO put this in somewhere else and integrate with layer offset


        # for node in self.graph.nodes:
        #     self.sub_memory_hex += float_list_to_byte_list([float(0)*self.out_feature_count], align=True, alignment=64)


        print('end of out messages',len(self.sub_memory_hex))
        self.out_messages_ptr = len(self.sub_memory_hex)  + self.memory_ptr
        out_ptr = self.out_messages_ptr

        return out_ptr

    # def map_out_messages(self):
    #     #    Set offset for next memory range
    #     self.offsets['out_messages'][0] = len(self.sub_memory_hex)
    #     #Assuming constant feature width
    #     size_messages = self.offsets['weights'][0]  -self.offsets['in_messages'] 

    #     self.offsets['out_messages'] = len(self.sub_memory_hex)

    # Dump
    # ===============================================

    # def dump_memory(self):
    #     with open(self.dump_file, 'w') as file:
    #         for i in range(len(self.sub_memory_hex)//64):
    #             file.write(''.join(self.sub_memory_hex[i*64:(i+1)*64]))
    #             file.write('\n')
    #         file.write(''.join(self.sub_memory_hex[64*(len(self.sub_memory_hex)//64):]))
    #         file.write('\n')


    def dump_memory(self, append_mode=False):
        mode = 'a' if append_mode else 'w'
        
        with open(self.dump_file, mode) as file:
            for i in range(len(self.sub_memory_hex)//64):
                if i > self.weight_addr_i:
                    addr = self.memory_ptr + 64*i
                    data = self.sub_memory_hex[i*64:(i+1)*64]
                file.write(''.join(self.sub_memory_hex[i*64:(i+1)*64]))
                file.write('\n')
            file.write(''.join(self.sub_memory_hex[64*(len(self.sub_memory_hex)//64):])) #TODO ensure no address gaps
            # file.write('\n')

    def pad_out_messages(self,eo_memory_ptr,out_messages_addr):
        padding_length = eo_memory_ptr - out_messages_addr
        padding = []
        while len(padding) < padding_length: 
            padding.append('00')
       

        with open(self.dump_file, 'a') as file:
            for i in range(len(padding)//64):
               
                file.write(''.join(padding[i*64:(i+1)*64]))
                file.write('\n')
            file.write(''.join(padding[64*(len(padding)//64):])) #TODO ensure no address gaps


    # def pad_list(self,input_list,target_length):
    #     current_length = len(input_list)
    #     if current_length < target_length:
    #         padding_length = target_length - current_length
    #         input_list.extend([0] * padding_length)
    #     return input_list
    

    def count_layers(self):
        count = 0
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear,GCNConv,GINConv,SAGEConv)):
                count += 1
        return count
    

    def contains_linear_layer(self):
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Linear):
                return True
        return False

         # def map_adj_list(self):
    #     #Dynamically change adj list between layers
    #     for idx,layer in enumerate(self.model.layers):
    #         # If there is a linear layer, add another adjacency list to point to the address of the features of a nodes own embedding
    #         #Need to find first edge layer to set adjacency list address



    #         #go thorugh each node in programming



    #         #Just need to map indexs of node inputs - offsets are provided in layer config
    #         #TODO: Make this simpler - adj list per layer? Make same for edge and node - or go over nodeslot programming dict
    #         if 'edge' in layer.name:
    #             #Make more concise
    #             if isinstance(layer, Linear):
    #                 for index, (u, v) in enumerate(self.graph.edges()):
    #                     print(f"Node: {self.graph[u][v]['meta']}")
    #                     if 'input' in layer.name:
    #                         self.graph[u][v]['meta']['adjacency_list_address'] = len(self.sub_memory_hex)
    #                     self.sub_memory_hex += int_list_to_byte_list(self.graph[u][v]['meta']['self_ptr'], align=True, alignment=64, pad_side="right")

    #             else:
    #                 for index, (u, v) in enumerate(self.graph.edges()):
    #                     if 'input' in layer.name:
    #                         self.graph[u][v]['meta']['adjacency_list_address'] = len(self.sub_memory_hex)
    #                     self.sub_memory_hex += int_list_to_byte_list(self.graph[u][v]['meta']['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right")

    #         else:
    #             if 'input' in layer.name:
    #                 for node in self.graph.nodes:
    #                     self.graph.nodes[node]["meta"]['adjacency_list_address'] = len(self.sub_memory_hex)



    #                     self.sub_memory_hex += int_list_to_byte_list(self.graph.nodes[node]["meta"]['self_ptr'], align=True, alignment=64, pad_side="right")



                        
    #                         if idx==0:
    #                 else:
    #                     aj_ptr = 0
    #                     for node in self.graph.nodes:
    #                             self.graph.nodes[node]["meta"]['adjacency_list_address'] = aj_ptr
    #                             #Change this
    #                             if isinstance(layer, Linear):

    #                             aj_ptr += len(int_list_to_byte_list(self.graph.nodes[node]["meta"]['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right"))
    #                             self.sub_memory_hex += int_list_to_byte_list(self.graph.nodes[node]["meta"]['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right")

    #         if(idx < self.num_layers-1):
    #             self.offsets['adj_list'][idx+1] = len(self.sub_memory_hex)


    #         # if isinstance(layer, Linear):
    #         #     #if edge layer,
    #         #     for node in programming:
                        
    #         #         if (idx ==0):
    #         #             #Setting address without layer offset
                        
    #         #             programming[node]["meta"]['adjacency_list_address'] = len(self.sub_memory_hex)
                
    #         #         self.sub_memory_hex += int_list_to_byte_list(programming[node]["meta"]['self_ptr'], align=True, alignment=64, pad_side="right")
    #         # else:
    #         #     for node in programming:
    #         #         if (idx ==0):
    #         #             programming[node]["meta"]['adjacency_list_address'] = len(self.sub_memory_hex)
                 
    #         #         self.sub_memory_hex += int_list_to_byte_list(programming[node]["meta"]['neighbour_message_ptrs'], align=True, alignment=64, pad_side="right")
            
           

    #     # Set offset for next memory range
    #     self.offsets['scale_factors'] = len(self.sub_memory_hex)