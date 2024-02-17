//
// Round robin arbiter
// Units that want to access a shared resource assert their bit in the 'request'
// bitmap. The arbiter picks a unit and sets the appropriate bit in the one hot
// signal grant_oh. This does not register grant_oh, which is valid the same
// cycle as the request. The update_lru signal indicates the granted unit has
// used the resource and should not receive access again until other requestors
// have had a turn.
//

module rr_arbiter #(
    parameter NUM_REQUESTERS = 4
) (
    input                                      clk,
    input                                      resetn,
    
    input        [NUM_REQUESTERS - 1:0]        request,
    input                                      update_lru,
    
    output logic [NUM_REQUESTERS - 1:0]        grant_oh,
    output logic [$clog2(NUM_REQUESTERS)-1:0]  grant_bin
);

// arbiter #
// (
//     .PORTS                 (NUM_REQUESTERS),
//     .ARB_TYPE_ROUND_ROBIN  (1)
// ) arbiter_i 
// (
//     .clk            (clk),
//     .rst            (!resetn),
//     .request        (request),
//     .acknowledge    (update_lru),
//     .grant          (grant_oh),
//     .grant_valid    (),
//     .grant_encoded  (grant_bin)
// );


always_comb begin
    grant_oh = '0;
    grant_bin = '0;

    for (int i = 0; i < NUM_REQUESTERS; i++) begin
        if (request[i]) begin
            grant_oh = ({{(NUM_REQUESTERS-1){1'b0}}, 1'b1} << i);
            grant_bin = i;
            break;
        end
    end
end

endmodule
