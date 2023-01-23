// Copyright 1986-2019 Xilinx, Inc. All Rights Reserved.
// --------------------------------------------------------------------------------
// Tool Version: Vivado v.2019.2.1 (lin64) Build 2729669 Thu Dec  5 04:48:12 MST 2019
// Date        : Sat Jan 21 21:49:19 2023
// Host        : ee-mill3.ee.ic.ac.uk running 64-bit CentOS Linux release 7.9.2009 (Core)
// Command     : write_verilog -force -mode synth_stub
//               /home/pg519/nfshome/fuzzy_carnival/ddr4_0_ex/ddr4_0_ex.srcs/sources_1/ip/axi_memory_interconnect/axi_memory_interconnect_stub.v
// Design      : axi_memory_interconnect
// Purpose     : Stub declaration of top-level module interface
// Device      : xcu250-figd2104-2L-e
// --------------------------------------------------------------------------------

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// The synthesis directives are for Synopsys Synplify support to prevent IO buffer insertion.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* X_CORE_INFO = "axi_interconnect_v1_7_17_top,Vivado 2019.2.1" *)
module axi_memory_interconnect(INTERCONNECT_ACLK, INTERCONNECT_ARESETN, 
  S00_AXI_ARESET_OUT_N, S00_AXI_ACLK, S00_AXI_AWID, S00_AXI_AWADDR, S00_AXI_AWLEN, 
  S00_AXI_AWSIZE, S00_AXI_AWBURST, S00_AXI_AWLOCK, S00_AXI_AWCACHE, S00_AXI_AWPROT, 
  S00_AXI_AWQOS, S00_AXI_AWVALID, S00_AXI_AWREADY, S00_AXI_WDATA, S00_AXI_WSTRB, 
  S00_AXI_WLAST, S00_AXI_WVALID, S00_AXI_WREADY, S00_AXI_BID, S00_AXI_BRESP, S00_AXI_BVALID, 
  S00_AXI_BREADY, S00_AXI_ARID, S00_AXI_ARADDR, S00_AXI_ARLEN, S00_AXI_ARSIZE, 
  S00_AXI_ARBURST, S00_AXI_ARLOCK, S00_AXI_ARCACHE, S00_AXI_ARPROT, S00_AXI_ARQOS, 
  S00_AXI_ARVALID, S00_AXI_ARREADY, S00_AXI_RID, S00_AXI_RDATA, S00_AXI_RRESP, S00_AXI_RLAST, 
  S00_AXI_RVALID, S00_AXI_RREADY, S01_AXI_ARESET_OUT_N, S01_AXI_ACLK, S01_AXI_AWID, 
  S01_AXI_AWADDR, S01_AXI_AWLEN, S01_AXI_AWSIZE, S01_AXI_AWBURST, S01_AXI_AWLOCK, 
  S01_AXI_AWCACHE, S01_AXI_AWPROT, S01_AXI_AWQOS, S01_AXI_AWVALID, S01_AXI_AWREADY, 
  S01_AXI_WDATA, S01_AXI_WSTRB, S01_AXI_WLAST, S01_AXI_WVALID, S01_AXI_WREADY, S01_AXI_BID, 
  S01_AXI_BRESP, S01_AXI_BVALID, S01_AXI_BREADY, S01_AXI_ARID, S01_AXI_ARADDR, S01_AXI_ARLEN, 
  S01_AXI_ARSIZE, S01_AXI_ARBURST, S01_AXI_ARLOCK, S01_AXI_ARCACHE, S01_AXI_ARPROT, 
  S01_AXI_ARQOS, S01_AXI_ARVALID, S01_AXI_ARREADY, S01_AXI_RID, S01_AXI_RDATA, S01_AXI_RRESP, 
  S01_AXI_RLAST, S01_AXI_RVALID, S01_AXI_RREADY, S02_AXI_ARESET_OUT_N, S02_AXI_ACLK, 
  S02_AXI_AWID, S02_AXI_AWADDR, S02_AXI_AWLEN, S02_AXI_AWSIZE, S02_AXI_AWBURST, 
  S02_AXI_AWLOCK, S02_AXI_AWCACHE, S02_AXI_AWPROT, S02_AXI_AWQOS, S02_AXI_AWVALID, 
  S02_AXI_AWREADY, S02_AXI_WDATA, S02_AXI_WSTRB, S02_AXI_WLAST, S02_AXI_WVALID, 
  S02_AXI_WREADY, S02_AXI_BID, S02_AXI_BRESP, S02_AXI_BVALID, S02_AXI_BREADY, S02_AXI_ARID, 
  S02_AXI_ARADDR, S02_AXI_ARLEN, S02_AXI_ARSIZE, S02_AXI_ARBURST, S02_AXI_ARLOCK, 
  S02_AXI_ARCACHE, S02_AXI_ARPROT, S02_AXI_ARQOS, S02_AXI_ARVALID, S02_AXI_ARREADY, 
  S02_AXI_RID, S02_AXI_RDATA, S02_AXI_RRESP, S02_AXI_RLAST, S02_AXI_RVALID, S02_AXI_RREADY, 
  S03_AXI_ARESET_OUT_N, S03_AXI_ACLK, S03_AXI_AWID, S03_AXI_AWADDR, S03_AXI_AWLEN, 
  S03_AXI_AWSIZE, S03_AXI_AWBURST, S03_AXI_AWLOCK, S03_AXI_AWCACHE, S03_AXI_AWPROT, 
  S03_AXI_AWQOS, S03_AXI_AWVALID, S03_AXI_AWREADY, S03_AXI_WDATA, S03_AXI_WSTRB, 
  S03_AXI_WLAST, S03_AXI_WVALID, S03_AXI_WREADY, S03_AXI_BID, S03_AXI_BRESP, S03_AXI_BVALID, 
  S03_AXI_BREADY, S03_AXI_ARID, S03_AXI_ARADDR, S03_AXI_ARLEN, S03_AXI_ARSIZE, 
  S03_AXI_ARBURST, S03_AXI_ARLOCK, S03_AXI_ARCACHE, S03_AXI_ARPROT, S03_AXI_ARQOS, 
  S03_AXI_ARVALID, S03_AXI_ARREADY, S03_AXI_RID, S03_AXI_RDATA, S03_AXI_RRESP, S03_AXI_RLAST, 
  S03_AXI_RVALID, S03_AXI_RREADY, M00_AXI_ARESET_OUT_N, M00_AXI_ACLK, M00_AXI_AWID, 
  M00_AXI_AWADDR, M00_AXI_AWLEN, M00_AXI_AWSIZE, M00_AXI_AWBURST, M00_AXI_AWLOCK, 
  M00_AXI_AWCACHE, M00_AXI_AWPROT, M00_AXI_AWQOS, M00_AXI_AWVALID, M00_AXI_AWREADY, 
  M00_AXI_WDATA, M00_AXI_WSTRB, M00_AXI_WLAST, M00_AXI_WVALID, M00_AXI_WREADY, M00_AXI_BID, 
  M00_AXI_BRESP, M00_AXI_BVALID, M00_AXI_BREADY, M00_AXI_ARID, M00_AXI_ARADDR, M00_AXI_ARLEN, 
  M00_AXI_ARSIZE, M00_AXI_ARBURST, M00_AXI_ARLOCK, M00_AXI_ARCACHE, M00_AXI_ARPROT, 
  M00_AXI_ARQOS, M00_AXI_ARVALID, M00_AXI_ARREADY, M00_AXI_RID, M00_AXI_RDATA, M00_AXI_RRESP, 
  M00_AXI_RLAST, M00_AXI_RVALID, M00_AXI_RREADY)
/* synthesis syn_black_box black_box_pad_pin="INTERCONNECT_ACLK,INTERCONNECT_ARESETN,S00_AXI_ARESET_OUT_N,S00_AXI_ACLK,S00_AXI_AWID[0:0],S00_AXI_AWADDR[33:0],S00_AXI_AWLEN[7:0],S00_AXI_AWSIZE[2:0],S00_AXI_AWBURST[1:0],S00_AXI_AWLOCK,S00_AXI_AWCACHE[3:0],S00_AXI_AWPROT[2:0],S00_AXI_AWQOS[3:0],S00_AXI_AWVALID,S00_AXI_AWREADY,S00_AXI_WDATA[511:0],S00_AXI_WSTRB[63:0],S00_AXI_WLAST,S00_AXI_WVALID,S00_AXI_WREADY,S00_AXI_BID[0:0],S00_AXI_BRESP[1:0],S00_AXI_BVALID,S00_AXI_BREADY,S00_AXI_ARID[0:0],S00_AXI_ARADDR[33:0],S00_AXI_ARLEN[7:0],S00_AXI_ARSIZE[2:0],S00_AXI_ARBURST[1:0],S00_AXI_ARLOCK,S00_AXI_ARCACHE[3:0],S00_AXI_ARPROT[2:0],S00_AXI_ARQOS[3:0],S00_AXI_ARVALID,S00_AXI_ARREADY,S00_AXI_RID[0:0],S00_AXI_RDATA[511:0],S00_AXI_RRESP[1:0],S00_AXI_RLAST,S00_AXI_RVALID,S00_AXI_RREADY,S01_AXI_ARESET_OUT_N,S01_AXI_ACLK,S01_AXI_AWID[0:0],S01_AXI_AWADDR[33:0],S01_AXI_AWLEN[7:0],S01_AXI_AWSIZE[2:0],S01_AXI_AWBURST[1:0],S01_AXI_AWLOCK,S01_AXI_AWCACHE[3:0],S01_AXI_AWPROT[2:0],S01_AXI_AWQOS[3:0],S01_AXI_AWVALID,S01_AXI_AWREADY,S01_AXI_WDATA[511:0],S01_AXI_WSTRB[63:0],S01_AXI_WLAST,S01_AXI_WVALID,S01_AXI_WREADY,S01_AXI_BID[0:0],S01_AXI_BRESP[1:0],S01_AXI_BVALID,S01_AXI_BREADY,S01_AXI_ARID[0:0],S01_AXI_ARADDR[33:0],S01_AXI_ARLEN[7:0],S01_AXI_ARSIZE[2:0],S01_AXI_ARBURST[1:0],S01_AXI_ARLOCK,S01_AXI_ARCACHE[3:0],S01_AXI_ARPROT[2:0],S01_AXI_ARQOS[3:0],S01_AXI_ARVALID,S01_AXI_ARREADY,S01_AXI_RID[0:0],S01_AXI_RDATA[511:0],S01_AXI_RRESP[1:0],S01_AXI_RLAST,S01_AXI_RVALID,S01_AXI_RREADY,S02_AXI_ARESET_OUT_N,S02_AXI_ACLK,S02_AXI_AWID[0:0],S02_AXI_AWADDR[33:0],S02_AXI_AWLEN[7:0],S02_AXI_AWSIZE[2:0],S02_AXI_AWBURST[1:0],S02_AXI_AWLOCK,S02_AXI_AWCACHE[3:0],S02_AXI_AWPROT[2:0],S02_AXI_AWQOS[3:0],S02_AXI_AWVALID,S02_AXI_AWREADY,S02_AXI_WDATA[511:0],S02_AXI_WSTRB[63:0],S02_AXI_WLAST,S02_AXI_WVALID,S02_AXI_WREADY,S02_AXI_BID[0:0],S02_AXI_BRESP[1:0],S02_AXI_BVALID,S02_AXI_BREADY,S02_AXI_ARID[0:0],S02_AXI_ARADDR[33:0],S02_AXI_ARLEN[7:0],S02_AXI_ARSIZE[2:0],S02_AXI_ARBURST[1:0],S02_AXI_ARLOCK,S02_AXI_ARCACHE[3:0],S02_AXI_ARPROT[2:0],S02_AXI_ARQOS[3:0],S02_AXI_ARVALID,S02_AXI_ARREADY,S02_AXI_RID[0:0],S02_AXI_RDATA[511:0],S02_AXI_RRESP[1:0],S02_AXI_RLAST,S02_AXI_RVALID,S02_AXI_RREADY,S03_AXI_ARESET_OUT_N,S03_AXI_ACLK,S03_AXI_AWID[0:0],S03_AXI_AWADDR[33:0],S03_AXI_AWLEN[7:0],S03_AXI_AWSIZE[2:0],S03_AXI_AWBURST[1:0],S03_AXI_AWLOCK,S03_AXI_AWCACHE[3:0],S03_AXI_AWPROT[2:0],S03_AXI_AWQOS[3:0],S03_AXI_AWVALID,S03_AXI_AWREADY,S03_AXI_WDATA[511:0],S03_AXI_WSTRB[63:0],S03_AXI_WLAST,S03_AXI_WVALID,S03_AXI_WREADY,S03_AXI_BID[0:0],S03_AXI_BRESP[1:0],S03_AXI_BVALID,S03_AXI_BREADY,S03_AXI_ARID[0:0],S03_AXI_ARADDR[33:0],S03_AXI_ARLEN[7:0],S03_AXI_ARSIZE[2:0],S03_AXI_ARBURST[1:0],S03_AXI_ARLOCK,S03_AXI_ARCACHE[3:0],S03_AXI_ARPROT[2:0],S03_AXI_ARQOS[3:0],S03_AXI_ARVALID,S03_AXI_ARREADY,S03_AXI_RID[0:0],S03_AXI_RDATA[511:0],S03_AXI_RRESP[1:0],S03_AXI_RLAST,S03_AXI_RVALID,S03_AXI_RREADY,M00_AXI_ARESET_OUT_N,M00_AXI_ACLK,M00_AXI_AWID[3:0],M00_AXI_AWADDR[33:0],M00_AXI_AWLEN[7:0],M00_AXI_AWSIZE[2:0],M00_AXI_AWBURST[1:0],M00_AXI_AWLOCK,M00_AXI_AWCACHE[3:0],M00_AXI_AWPROT[2:0],M00_AXI_AWQOS[3:0],M00_AXI_AWVALID,M00_AXI_AWREADY,M00_AXI_WDATA[511:0],M00_AXI_WSTRB[63:0],M00_AXI_WLAST,M00_AXI_WVALID,M00_AXI_WREADY,M00_AXI_BID[3:0],M00_AXI_BRESP[1:0],M00_AXI_BVALID,M00_AXI_BREADY,M00_AXI_ARID[3:0],M00_AXI_ARADDR[33:0],M00_AXI_ARLEN[7:0],M00_AXI_ARSIZE[2:0],M00_AXI_ARBURST[1:0],M00_AXI_ARLOCK,M00_AXI_ARCACHE[3:0],M00_AXI_ARPROT[2:0],M00_AXI_ARQOS[3:0],M00_AXI_ARVALID,M00_AXI_ARREADY,M00_AXI_RID[3:0],M00_AXI_RDATA[511:0],M00_AXI_RRESP[1:0],M00_AXI_RLAST,M00_AXI_RVALID,M00_AXI_RREADY" */;
  input INTERCONNECT_ACLK;
  input INTERCONNECT_ARESETN;
  output S00_AXI_ARESET_OUT_N;
  input S00_AXI_ACLK;
  input [0:0]S00_AXI_AWID;
  input [33:0]S00_AXI_AWADDR;
  input [7:0]S00_AXI_AWLEN;
  input [2:0]S00_AXI_AWSIZE;
  input [1:0]S00_AXI_AWBURST;
  input S00_AXI_AWLOCK;
  input [3:0]S00_AXI_AWCACHE;
  input [2:0]S00_AXI_AWPROT;
  input [3:0]S00_AXI_AWQOS;
  input S00_AXI_AWVALID;
  output S00_AXI_AWREADY;
  input [511:0]S00_AXI_WDATA;
  input [63:0]S00_AXI_WSTRB;
  input S00_AXI_WLAST;
  input S00_AXI_WVALID;
  output S00_AXI_WREADY;
  output [0:0]S00_AXI_BID;
  output [1:0]S00_AXI_BRESP;
  output S00_AXI_BVALID;
  input S00_AXI_BREADY;
  input [0:0]S00_AXI_ARID;
  input [33:0]S00_AXI_ARADDR;
  input [7:0]S00_AXI_ARLEN;
  input [2:0]S00_AXI_ARSIZE;
  input [1:0]S00_AXI_ARBURST;
  input S00_AXI_ARLOCK;
  input [3:0]S00_AXI_ARCACHE;
  input [2:0]S00_AXI_ARPROT;
  input [3:0]S00_AXI_ARQOS;
  input S00_AXI_ARVALID;
  output S00_AXI_ARREADY;
  output [0:0]S00_AXI_RID;
  output [511:0]S00_AXI_RDATA;
  output [1:0]S00_AXI_RRESP;
  output S00_AXI_RLAST;
  output S00_AXI_RVALID;
  input S00_AXI_RREADY;
  output S01_AXI_ARESET_OUT_N;
  input S01_AXI_ACLK;
  input [0:0]S01_AXI_AWID;
  input [33:0]S01_AXI_AWADDR;
  input [7:0]S01_AXI_AWLEN;
  input [2:0]S01_AXI_AWSIZE;
  input [1:0]S01_AXI_AWBURST;
  input S01_AXI_AWLOCK;
  input [3:0]S01_AXI_AWCACHE;
  input [2:0]S01_AXI_AWPROT;
  input [3:0]S01_AXI_AWQOS;
  input S01_AXI_AWVALID;
  output S01_AXI_AWREADY;
  input [511:0]S01_AXI_WDATA;
  input [63:0]S01_AXI_WSTRB;
  input S01_AXI_WLAST;
  input S01_AXI_WVALID;
  output S01_AXI_WREADY;
  output [0:0]S01_AXI_BID;
  output [1:0]S01_AXI_BRESP;
  output S01_AXI_BVALID;
  input S01_AXI_BREADY;
  input [0:0]S01_AXI_ARID;
  input [33:0]S01_AXI_ARADDR;
  input [7:0]S01_AXI_ARLEN;
  input [2:0]S01_AXI_ARSIZE;
  input [1:0]S01_AXI_ARBURST;
  input S01_AXI_ARLOCK;
  input [3:0]S01_AXI_ARCACHE;
  input [2:0]S01_AXI_ARPROT;
  input [3:0]S01_AXI_ARQOS;
  input S01_AXI_ARVALID;
  output S01_AXI_ARREADY;
  output [0:0]S01_AXI_RID;
  output [511:0]S01_AXI_RDATA;
  output [1:0]S01_AXI_RRESP;
  output S01_AXI_RLAST;
  output S01_AXI_RVALID;
  input S01_AXI_RREADY;
  output S02_AXI_ARESET_OUT_N;
  input S02_AXI_ACLK;
  input [0:0]S02_AXI_AWID;
  input [33:0]S02_AXI_AWADDR;
  input [7:0]S02_AXI_AWLEN;
  input [2:0]S02_AXI_AWSIZE;
  input [1:0]S02_AXI_AWBURST;
  input S02_AXI_AWLOCK;
  input [3:0]S02_AXI_AWCACHE;
  input [2:0]S02_AXI_AWPROT;
  input [3:0]S02_AXI_AWQOS;
  input S02_AXI_AWVALID;
  output S02_AXI_AWREADY;
  input [511:0]S02_AXI_WDATA;
  input [63:0]S02_AXI_WSTRB;
  input S02_AXI_WLAST;
  input S02_AXI_WVALID;
  output S02_AXI_WREADY;
  output [0:0]S02_AXI_BID;
  output [1:0]S02_AXI_BRESP;
  output S02_AXI_BVALID;
  input S02_AXI_BREADY;
  input [0:0]S02_AXI_ARID;
  input [33:0]S02_AXI_ARADDR;
  input [7:0]S02_AXI_ARLEN;
  input [2:0]S02_AXI_ARSIZE;
  input [1:0]S02_AXI_ARBURST;
  input S02_AXI_ARLOCK;
  input [3:0]S02_AXI_ARCACHE;
  input [2:0]S02_AXI_ARPROT;
  input [3:0]S02_AXI_ARQOS;
  input S02_AXI_ARVALID;
  output S02_AXI_ARREADY;
  output [0:0]S02_AXI_RID;
  output [511:0]S02_AXI_RDATA;
  output [1:0]S02_AXI_RRESP;
  output S02_AXI_RLAST;
  output S02_AXI_RVALID;
  input S02_AXI_RREADY;
  output S03_AXI_ARESET_OUT_N;
  input S03_AXI_ACLK;
  input [0:0]S03_AXI_AWID;
  input [33:0]S03_AXI_AWADDR;
  input [7:0]S03_AXI_AWLEN;
  input [2:0]S03_AXI_AWSIZE;
  input [1:0]S03_AXI_AWBURST;
  input S03_AXI_AWLOCK;
  input [3:0]S03_AXI_AWCACHE;
  input [2:0]S03_AXI_AWPROT;
  input [3:0]S03_AXI_AWQOS;
  input S03_AXI_AWVALID;
  output S03_AXI_AWREADY;
  input [511:0]S03_AXI_WDATA;
  input [63:0]S03_AXI_WSTRB;
  input S03_AXI_WLAST;
  input S03_AXI_WVALID;
  output S03_AXI_WREADY;
  output [0:0]S03_AXI_BID;
  output [1:0]S03_AXI_BRESP;
  output S03_AXI_BVALID;
  input S03_AXI_BREADY;
  input [0:0]S03_AXI_ARID;
  input [33:0]S03_AXI_ARADDR;
  input [7:0]S03_AXI_ARLEN;
  input [2:0]S03_AXI_ARSIZE;
  input [1:0]S03_AXI_ARBURST;
  input S03_AXI_ARLOCK;
  input [3:0]S03_AXI_ARCACHE;
  input [2:0]S03_AXI_ARPROT;
  input [3:0]S03_AXI_ARQOS;
  input S03_AXI_ARVALID;
  output S03_AXI_ARREADY;
  output [0:0]S03_AXI_RID;
  output [511:0]S03_AXI_RDATA;
  output [1:0]S03_AXI_RRESP;
  output S03_AXI_RLAST;
  output S03_AXI_RVALID;
  input S03_AXI_RREADY;
  output M00_AXI_ARESET_OUT_N;
  input M00_AXI_ACLK;
  output [3:0]M00_AXI_AWID;
  output [33:0]M00_AXI_AWADDR;
  output [7:0]M00_AXI_AWLEN;
  output [2:0]M00_AXI_AWSIZE;
  output [1:0]M00_AXI_AWBURST;
  output M00_AXI_AWLOCK;
  output [3:0]M00_AXI_AWCACHE;
  output [2:0]M00_AXI_AWPROT;
  output [3:0]M00_AXI_AWQOS;
  output M00_AXI_AWVALID;
  input M00_AXI_AWREADY;
  output [511:0]M00_AXI_WDATA;
  output [63:0]M00_AXI_WSTRB;
  output M00_AXI_WLAST;
  output M00_AXI_WVALID;
  input M00_AXI_WREADY;
  input [3:0]M00_AXI_BID;
  input [1:0]M00_AXI_BRESP;
  input M00_AXI_BVALID;
  output M00_AXI_BREADY;
  output [3:0]M00_AXI_ARID;
  output [33:0]M00_AXI_ARADDR;
  output [7:0]M00_AXI_ARLEN;
  output [2:0]M00_AXI_ARSIZE;
  output [1:0]M00_AXI_ARBURST;
  output M00_AXI_ARLOCK;
  output [3:0]M00_AXI_ARCACHE;
  output [2:0]M00_AXI_ARPROT;
  output [3:0]M00_AXI_ARQOS;
  output M00_AXI_ARVALID;
  input M00_AXI_ARREADY;
  input [3:0]M00_AXI_RID;
  input [511:0]M00_AXI_RDATA;
  input [1:0]M00_AXI_RRESP;
  input M00_AXI_RLAST;
  input M00_AXI_RVALID;
  output M00_AXI_RREADY;
endmodule