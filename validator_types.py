from dataclasses import dataclass


@dataclass
class IdentifierMapEntry:
    new_name: str
    block_id: bool
    has_linkage: bool

class IdentifierAttrs:
    pass

@dataclass
class FunAttr(IdentifierAttrs):
    is_defined: bool
    is_global: bool

@dataclass
class StaticAttr(IdentifierAttrs):
    init: "InitialValue"
    is_global: bool

class LocalAttr(IdentifierAttrs):
    pass

class InitialValue:
    pass

class InitialValueTentative(InitialValue):
    pass

@dataclass
class InitialValueInt(InitialValue):
    value: int

class InitialValueNoInitializer(InitialValue):
    pass

@dataclass
class SymbolsTableItem:
    name: str
    type: str
    attrs: IdentifierAttrs