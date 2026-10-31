"""Small Flow authoring helper; owns no architecture instructions or timing."""
from ....ir.common import Event, SlotHandle, TileValue
from ....ir.flow import FlowOp


class FlowBuilder:
    def __init__(self):
        self.ops=[]
        self.producer="producer"
        self.generations={}
        self.release_events={}

    def emit(self, name, capability, executor, *, inputs=(), reads=(), writes=(),
             slot=None, semantic='', operands=(), shape=None, dtype='fp32'):
        phase='ready' if capability in ('load_async','acquire') else 'complete'
        if capability=='release':phase='released'
        event=Event(name+'.'+phase,name,phase)
        values=() if shape is None else (TileValue(name+'.value',name,shape,dtype),)
        self.ops.append(FlowOp(name,capability,executor,reads,writes,tuple(inputs),(event,),slot,
                               semantic,tuple(operands),values))
        return event.id, values[0].id if values else None

    def acquire(self, name, buffer, index):
        key=(buffer,index)
        slot=SlotHandle(buffer,index,self.generations.get(key,0))
        inputs=(self.release_events[key],) if key in self.release_events else ()
        event,_=self.emit(name,'acquire',self.producer,slot=slot,inputs=inputs)
        return slot,event

    def release(self, name, slot, consumer_events):
        event,_=self.emit(name,'release',self.producer,slot=slot,inputs=consumer_events)
        key=(slot.buffer,slot.slot)
        self.generations[key]=slot.generation+1
        self.release_events[key]=event
