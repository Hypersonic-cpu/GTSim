# Text and legacy Qwen DAG differences

This document records the **initial four-run comparison**, before the requested
IR-only synchronization/precision correction. The current library changes
Small-M prior-copy ordering to Issue and restores legacy Decode input-conversion,
BF16 probability and K-buffer-reuse semantics. Corrected results are reported
separately in `issue-aligned-results.md`; the original evidence is retained.

This comparison uses the current `serve-sim` legacy generator, not a historical
report's DAG. Both inputs contain one complete Qwen3-30B-A3B MoE decode block.
`b1.json` uses four KV splits and direct experts; `b16.json` uses non-split
attention and grouped experts. Both versions consume identical context lengths
and deterministic routing IDs/probabilities. Model dimensions, BF16 tensors,
valid regions, selected experts and residual identities agree.

## Compiler organization

The text path parses actual ProgIR/AlgoIR/FlowIR source into the independent
`src/ast` representations, composes shared library regions, lowers structured
FlowIR to ArchIR and expands native operations at the streaming boundary. Static
parameters, branches and loops are resolved at that final stage. The Qwen
Program contains composition and tensor bindings rather than instruction loops.

The comparison's legacy path uses `gtsim_frontend`'s current Qwen Program builder,
algorithm instantiation and Hopper/Execution lowerers. These imports are confined
to comparison/benchmark tools; the new parser/compiler does not call them.
Both executable graphs are interpreted by the same C++ DAG execution core.

## Matrix kernels

The five projection/expert GEMMs in each case agree in normalized native opcode
counts, fragment shapes, warp mapping, valid memory regions and CTA resources.
The physical Small-M tile has 16 rows; masked rows do not become logical tokens
or generate padded global input/output accesses. Expert weight identities and
token/slot mappings remain distinct.

Their dependency graphs differ: the text Small-M schedule waits for the previous
K-stage copy batch's **Complete** event before issuing the next copy batch; the
legacy variant uses its **Issue** event. Both retain consumer waits before buffer
reuse. Complete ordering is more conservative and can reduce memory/compute
overlap even when every instruction and byte count agrees. This comparison does
not establish that the more conservative schedule is required for correctness.

B=1 has eight assignments to eight experts, 128 physical rows and 120 padding
rows (6.25% row utilization). B=16 has 128 assignments to 76 experts, 1,216
physical rows and 1,088 padding rows (10.53%). The grouped implementation is a
grouped Small-M MMA baseline; these results do not establish calibrated grouped
GEMM accuracy or an optimal dispatch policy.

## Decode attention and combination

Both graphs use one real query token, 32 query heads, four KV heads, D=128,
scaled QK reduction and stable online softmax. Query heads map eight-to-one to
KV heads. Each request includes its newly appended token and only its own valid
history. Split-KV writes partial output and LSE statistics and has a separate
stable LSE combination kernel.

The text schedule explicitly converts BF16 Q/K/V to FP32 and retains FP32
softmax probabilities for SIMT PV reduction. The legacy schedule has fewer
explicit input casts and converts the probabilities to BF16. Thus they express
the same attention formula with different intermediate precision and cost graphs;
bitwise numerical equality is not claimed. Decode node counts are 27,136 versus
25,728 at B=1 and 856,448 versus 813,344 at B=16 (text versus legacy).

The new split combination loads each split's Partial and LSE rows separately;
legacy combines those rows into larger transfers. With four splits, that is 256
versus 64 load nodes across 32 heads, while logical bytes agree. Combination
graphs have 576 versus 384 nodes. The text combination uses the generic vector
resource profile (16,384 shared bytes, 32,768 register bytes per CTA); legacy
uses its four-row decode profile (2,304 and 16,912). Decode itself has the same
131,328 shared bytes per CTA but different register estimates (32,768 versus
17,920). Resource and transfer granularity changes also influence timing.

## Auxiliary operations

| Operator        | Text path                                                                            | Legacy path                                                 | Consequence                                                                                    |
| --------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| RMSNorm         | Divide FP32 squared sum by width; immediate epsilon                                  | Multiply by a precomputed reciprocal and fill constants     | 14 versus 16 nodes per row; same normalization formula and learned scale                       |
| Q/K norm + RoPE | Whole-head half-layout rotation, explicit position/theta operands and frequency work | Half-vector arithmetic and stores, more constant/fill nodes | 25 versus 33 nodes per head; V remains unnormalized/unrotated                                  |
| Router Top-K    | Separate ID/probability packing and explicit masking recurrence                      | Different mask/packing instruction decomposition            | 40 versus 39 nodes per token; softmax, Top-8 and selected-probability renormalization retained |
| Histogram/scan  | One route-ID load per token                                                          | One bulk route-ID load                                      | At B=16, 37 versus 22 nodes; equal logical traffic                                             |
| Expert gather   | Static token/slot/expert metadata                                                    | Explicit metadata fills/packing                             | At B=16, 1,408 versus 1,920 nodes; equal gathered data regions                                 |
| SwiGLU          | Immediate constants; distinct Gate/Up subregions                                     | Separate constant fills; distinct Gate/Up subregions        | 96 versus 112 nodes at B=1, 1,536 versus 1,792 at B=16                                         |

RoPE uses request-specific decode positions, theta=1,000,000 and half-channel
pairing. Native immediate metadata records the operands even though GTSim is a
timing simulator. Some instruction names denote cost classes: a trigonometric
cost opcode may cover both sine and cosine; the mathematical algorithm retains
the actual function. This is not numerical execution of model activations.

## Allocation and traffic

The text version packs expert weights into expert-indexed tensors and KV into
request/head regions. Legacy uses separate active-expert and request/head
allocations. Text weight allocation reserves all 128 experts but executes reads
only for selected experts. Reserving addresses incurs no synthetic HBM transfer.

All 37 kernel comparisons have equal valid logical read/write coverage after
normalizing these layouts. Physical base addresses, alignment and cache-line/set
mapping can differ. Equal logical bytes therefore do not imply equal cache hit
rates, physical transaction counts or actual HBM bytes. The simulation report
records actual traffic separately from this structural equivalence.

## Interpretation

Total nodes are 589,458 versus 588,165 at B=1 and 4,604,999 versus 4,567,304 at
B=16. Node count alone cannot predict performance: dependencies, resources,
transaction packetization and physical addresses also matter. The four runs vary
several of these together, so individual cycle deltas are observations rather
than causal isolation experiments.

Independent CPU double-precision oracles check attention recurrence, split LSE,
RMSNorm, RoPE and weighted expert mapping. Structured address audits check the
request/head/split/expert identities. These checks establish mathematical and
graph properties, not measured H100 latency accuracy. New decode/vector/grouped
timings remain uncalibrated. The separate full-query FA3 text-library port does
not establish graph/cycle parity with the original calibrated full-tile profile.

The runtime runs every kernel on one persistent GPU/MemorySystem/L2 and one
ordered stream. Reported stream cycles include memory draining. Four-run launch
overhead is explicitly zero. Python expansion, JSON pipe transport and C++ graph
parsing take host wall time but do not advance the simulated GPU clock.
