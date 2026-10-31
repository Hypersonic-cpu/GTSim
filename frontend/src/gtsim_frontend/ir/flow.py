from dataclasses import dataclass
from .common import Header, Tensor, TensorRegion, SlotHandle, Event, TileValue, require
from .mapping import Executor, Buffer, Mapping


@dataclass(frozen=True)
class FlowOp:
    id: str
    capability: str
    executor: str
    reads: tuple[TensorRegion, ...] = ()
    writes: tuple[TensorRegion, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[Event, ...] = ()
    slot: SlotHandle | None = None
    semantic: str = ""
    operands: tuple[str, ...] = ()
    results: tuple[TileValue, ...] = ()


@dataclass(frozen=True)
class FlowKernel(Header):
    family: str = "gemm"
    schedule_id: str = ""
    tensors: tuple[Tensor, ...] = ()
    executors: tuple[Executor, ...] = ()
    buffers: tuple[Buffer, ...] = ()
    mapping: Mapping = Mapping()
    ops: tuple[FlowOp, ...] = ()

    def verify(self):
        super().verify()
        require(self.family in ("gemm","fa3") and bool(self.schedule_id), self.symbol, "unknown family or missing explicit schedule")
        tensors = {t.id:t for t in self.tensors}
        require(len(tensors) == len(self.tensors),self.symbol,"duplicate tensor")
        for t in self.tensors:t.verify()
        require(len({e.id for e in self.executors}) == len(self.executors),self.symbol,"duplicate executor")
        require(len({b.id for b in self.buffers}) == len(self.buffers),self.symbol,"duplicate buffer")
        self.mapping.verify(self.executors,self.buffers)
        executors = {e.id for e in self.executors}
        buffers = {b.id:b for b in self.buffers}
        events, ids, owners, generations, values = {}, set(), {}, {}, {}
        value_slots, consumers = {}, {}
        for op in self.ops:
            require(op.id not in ids,op.id,"duplicate operation")
            ids.add(op.id)
            require(op.executor in executors,op.id,"unknown executor")
            require(op.capability in ("acquire","load_async","matmul","compute","store_async","release"),op.id,"unknown capability")
            require(all(e in events for e in op.inputs),op.id,"missing producer, forward event, or dependency cycle")
            require(all(v in values for v in op.operands),op.id,"missing SSA value producer")
            for operand in op.operands:
                require(any(events[e].producer == values[operand].producer for e in op.inputs),op.id,"SSA operand lacks readiness/completion dependency")
                if operand in value_slots:
                    slot_key=value_slots[operand]
                    require(owners.get(slot_key[:2]) == slot_key[2],op.id,"read after slot release or overwrite")
                    consumers.setdefault(slot_key,set()).update(e.id for e in op.outputs if e.phase == "complete")
            for value in op.results:
                value.verify()
                require(value.id not in values and value.producer == op.id,op.id,"duplicate value or invalid producer")
                values[value.id]=value
                if op.capability == "load_async" and op.slot:
                    value_slots[value.id]=(op.slot.buffer,op.slot.slot,op.slot.generation)
            for region in op.reads+op.writes:region.verify(tensors)
            require(all(not tensors[r.tensor].read_only for r in op.writes),op.id,"write to read-only tensor")
            if op.slot:
                s=op.slot; key=(s.buffer,s.slot)
                require(s.buffer in buffers and 0 <= s.slot < buffers[s.buffer].slots and s.generation >= 0,op.id,"invalid buffer slot")
                if op.capability == "acquire":
                    require(key not in owners,op.id,"slot reused before release")
                    require(s.generation == generations.get(key, 0),op.id,"slot generation mismatch")
                    owners[key]=s.generation
                else:
                    require(owners.get(key) == s.generation,op.id,"slot not owned by generation")
                    if op.capability == "release":
                        require(bool(op.inputs) and all(events[e].phase == "complete" for e in op.inputs),op.id,"release requires last consumer completion")
                        require(consumers.get((s.buffer,s.slot,s.generation),set()) <= set(op.inputs),op.id,"release before last consumer completion")
                        del owners[key]
                        generations[key] = s.generation + 1
            for e in op.outputs:
                e.verify()
                require(e.id not in events and e.producer == op.id,op.id,"invalid event producer or duplicate event")
                events[e.id]=e
        require(not owners,self.symbol,"unreleased buffer slots")
