// ViruxAI – Neuromorphic AI Engine
// Module: SNN Core – Verilog RTL (Top-Level)
//
// Integrates:
//   - Array of N_NEURON LIF neurons
//   - On-chip BRAM weight memory (N_IN × N_NEURON × 8-bit)
//   - Pipelined accumulate-and-fire engine
//   - Spike output bus
//
// Non-Von Neumann design: weight BRAM is co-located with the MAC unit
// so no external RAM traffic is required during inference.

`timescale 1ns / 1ps

module snn_core #(
    parameter N_IN      = 64,    // Number of input neurons
    parameter N_NEURON  = 32,    // Number of hidden neurons
    parameter W_BITS    = 8,     // Weight bitwidth
    parameter DATA_W    = 16
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [N_IN-1:0]         i_spikes,    // Input spike bus
    input  wire                    i_valid,
    output reg  [N_NEURON-1:0]     o_spikes,    // Output spike bus
    output reg                     o_valid
);

    // -------------------------------------------------------------------------
    // On-chip weight memory (In-Memory Computing)
    // -------------------------------------------------------------------------
    reg signed [W_BITS-1:0] weight_mem [0:N_NEURON-1][0:N_IN-1];
    integer init_i, init_j;
    initial begin
        for (init_i = 0; init_i < N_NEURON; init_i = init_i + 1)
            for (init_j = 0; init_j < N_IN; init_j = init_j + 1)
                weight_mem[init_i][init_j] = $random % 64;  // Simulated initialisation
    end

    // -------------------------------------------------------------------------
    // Accumulator: weighted sum of input spikes
    // -------------------------------------------------------------------------
    reg signed [DATA_W-1:0] accum [0:N_NEURON-1];
    integer n, k;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (n = 0; n < N_NEURON; n = n + 1)
                accum[n] <= 0;
            o_valid <= 1'b0;
        end else if (i_valid) begin
            for (n = 0; n < N_NEURON; n = n + 1) begin
                accum[n] <= 0;
                for (k = 0; k < N_IN; k = k + 1)
                    if (i_spikes[k])
                        accum[n] <= accum[n] + weight_mem[n][k];
            end
            o_valid <= 1'b1;
        end else begin
            o_valid <= 1'b0;
        end
    end

    // -------------------------------------------------------------------------
    // LIF neuron array
    // -------------------------------------------------------------------------
    genvar gn;
    generate
        for (gn = 0; gn < N_NEURON; gn = gn + 1) begin : lif_array
            lif_neuron #(
                .DATA_W (DATA_W)
            ) u_lif (
                .clk       (clk),
                .rst_n     (rst_n),
                .i_current (accum[gn]),
                .o_spike   (o_spikes[gn]),
                .o_vmem    ()
            );
        end
    endgenerate

endmodule
