// ViruxAI – Neuromorphic AI Engine
// Module: LIF Neuron – Verilog RTL Implementation
//
// Implements a single LIF neuron using integer (fixed-point) arithmetic.
// Weight storage is co-located with the compute logic to eliminate
// dependency on external RAM (In-Memory Computing architecture).
//
// Interface:
//   clk       – Synchronous clock
//   rst_n     – Active-low reset
//   i_current – Input current (Q8.7 fixed-point, 16-bit signed)
//   o_spike   – Spike output (1 = firing, 0 = silent)
//   o_vmem    – Current membrane potential (16-bit signed)

`timescale 1ns / 1ps

module lif_neuron #(
    parameter DATA_W  = 16,
    parameter V_REST  = -16'sd8960,  // -70.0 * 128  (Q8.7)
    parameter V_TH    = -16'sd7040,  // -55.0 * 128
    parameter V_RESET = -16'sd9600,  // -75.0 * 128
    parameter TAU_INV = 16'sd13,     // ≈ dt/tau * 128
    parameter R_MEM   = 16'sd10,
    parameter T_REF   = 4'd20        // Refractory steps
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire signed [DATA_W-1:0]      i_current,
    output reg                           o_spike,
    output reg  signed [DATA_W-1:0]      o_vmem
);

    // Internal registers
    reg signed [DATA_W-1:0] vmem;
    reg [3:0]               ref_cnt;   // Refractory counter

    // Euler forward: ΔV = (dt/tau) * (-(V - V_rest) + R * I)
    wire signed [2*DATA_W-1:0] leak_term;
    wire signed [2*DATA_W-1:0] input_term;
    wire signed [2*DATA_W-1:0] dv;

    assign leak_term  = TAU_INV * (-(vmem - V_REST));
    assign input_term = TAU_INV * (R_MEM * i_current);
    assign dv         = (leak_term + input_term) >>> 7;  // Q-point shift

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vmem    <= V_REST;
            o_spike <= 1'b0;
            ref_cnt <= 4'd0;
        end else begin
            o_spike <= 1'b0;

            if (ref_cnt > 0) begin
                // Absolute refractory period
                vmem    <= V_RESET;
                ref_cnt <= ref_cnt - 1;
            end else begin
                vmem <= vmem + dv[DATA_W-1:0];

                if (vmem >= V_TH) begin
                    o_spike <= 1'b1;
                    vmem    <= V_RESET;
                    ref_cnt <= T_REF;
                end
            end
        end
    end

    assign o_vmem = vmem;

endmodule
