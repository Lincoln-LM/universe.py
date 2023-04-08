"""Patching for numba jitclass boxing/unboxing"""

from numba.experimental.jitclass.boxing import (
    _box,
    _cache_specialized_box,
    _generate_getter,
    _generate_method,
    _generate_setter,
)
from numba.experimental.jitclass import boxing as boxing_module


def _specialize_box(typ):
    """
    Create a subclass of Box that is specialized to the jitclass.

    This function caches the result to avoid code bloat.
    """
    # Check cache
    if typ in _cache_specialized_box:
        return _cache_specialized_box[typ]
    dct = {
        "__slots__": (),
        "_numba_type_": typ,
        "__doc__": typ.class_type.class_doc,
    }
    # Inject attributes as class properties
    for field in typ.struct:
        getter = _generate_getter(field)
        setter = _generate_setter(field)
        dct[field] = property(getter, setter)
    # Inject properties as class properties
    for field, impdct in typ.jit_props.items():
        getter = None
        setter = None
        if "get" in impdct:
            getter = _generate_getter(field)
        if "set" in impdct:
            setter = _generate_setter(field)
        # get docstring from either the fget or fset
        imp = impdct.get("get") or impdct.get("set") or None
        doc = getattr(imp, "__doc__", None)
        dct[field] = property(getter, setter, doc=doc)
    # Inject methods as class members
    supported_dunders = {
        "__abs__",
        "__bool__",
        "__complex__",
        "__contains__",
        "__float__",
        "__getitem__",
        "__hash__",
        "__index__",
        "__int__",
        "__len__",
        "__setitem__",
        "__str__",
        "__eq__",
        "__ne__",
        "__ge__",
        "__gt__",
        "__le__",
        "__lt__",
        "__add__",
        "__floordiv__",
        "__lshift__",
        "__mod__",
        "__mul__",
        "__neg__",
        "__pos__",
        "__pow__",
        "__rshift__",
        "__sub__",
        "__truediv__",
        "__and__",
        "__or__",
        "__xor__",
        "__iadd__",
        "__ifloordiv__",
        "__ilshift__",
        "__imod__",
        "__imul__",
        "__ipow__",
        "__irshift__",
        "__isub__",
        "__itruediv__",
        "__iand__",
        "__ior__",
        "__ixor__",
    }
    for name, func in typ.methods.items():
        if name == "__init__":
            continue
        if (
            name.startswith("__")
            and name.endswith("__")
            and name not in supported_dunders
        ):
            raise TypeError(f"Method '{name}' is not supported.")
        # only generate methods for njit functions
        dct[name] = (
            _generate_method(name, func) if getattr(func, "njit", True) else func
        )

    # Inject static methods as class members
    for name, func in typ.static_methods.items():
        dct[name] = _generate_method(name, func)

    # Create subclass
    subcls = type(typ.classname, (_box.Box,), dct)
    # Store to cache
    _cache_specialized_box[typ] = subcls

    # Pre-compile attribute getter.
    # Note: This must be done after the "box" class is created because
    #       compiling the getter requires the "box" class to be defined.
    for k, v in dct.items():
        if isinstance(v, property):
            prop = getattr(subcls, k)
            if prop.fget is not None:
                fget = prop.fget
                fast_fget = fget.compile((typ,))
                fget.disable_compile()
                setattr(
                    subcls,
                    k,
                    property(fast_fget, prop.fset, prop.fdel, doc=prop.__doc__),
                )

    return subcls


# patch function

boxing_module._specialize_box = _specialize_box
