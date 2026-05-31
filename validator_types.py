from dataclasses import dataclass

from typing import Optional

from parser_types import TypeNode


@dataclass
class IdentifierMapEntry:
    new_name: str
    block_id: int
    type: Optional[TypeNode]
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

@dataclass
class InitialValueLong(InitialValue):
    value: int

class InitialValueNoInitializer(InitialValue):
    pass

@dataclass
class SymbolsTableItem_LEGACY:
    name: str
    type: str
    attrs: IdentifierAttrs

@dataclass
class SymbolsTableItem:
    name: str
    type: TypeNode
    attrs: IdentifierAttrs